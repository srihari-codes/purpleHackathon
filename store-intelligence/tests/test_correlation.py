"""
tests/test_correlation.py — Unit tests for CorrelationEngine (app/correlation.py).

14 tests covering:
  - load_csv standard format
  - load_csv production (Brigade Bangalore) format
  - add_transaction
  - correlate within 5-min window → converted
  - correlate just outside window → not converted
  - correlate staff session → skipped
  - conversion_rate 0%, 100%, partial
  - is_converted before/after correlate
  - idempotent correlate (double run same result)
  - clear()
  - transaction_count
"""

import os
import tempfile

import pytest

from app.correlation import BILLING_WINDOW_SEC, CorrelationEngine, POSTransaction
from app.models import EventType, QueueEvent, VisitorSession
from tests.conftest import make_ts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_session(
    visitor_id: str,
    *,
    store_id: str = "STORE_TEST",
    has_billing_queue: bool = True,
    queue_offset: int = 0,
    is_staff: bool = False,
) -> VisitorSession:
    s = VisitorSession(
        visitor_id=visitor_id,
        store_id=store_id,
        start_time=make_ts(0),
        is_staff=is_staff,
    )
    if has_billing_queue:
        s.queue_events.append(QueueEvent(
            event_type=EventType.BILLING_QUEUE_JOIN,
            timestamp=make_ts(queue_offset),
        ))
    return s


def add_txn(engine: CorrelationEngine, store_id: str, offset: int):
    engine.add_transaction(POSTransaction(
        store_id=store_id,
        transaction_id=f"TXN_{offset}",
        timestamp=make_ts(offset),
    ))


# ---------------------------------------------------------------------------
# 1. load_csv — standard format
# ---------------------------------------------------------------------------

def test_load_csv_standard_format(correlation):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("store_id,transaction_id,timestamp,basket_value_inr\n")
        f.write(f"STORE_A,TXN_001,{make_ts(0)},250.00\n")
        f.write(f"STORE_A,TXN_002,{make_ts(60)},180.50\n")
        path = f.name
    try:
        loaded = correlation.load_csv(path)
        assert loaded == 2
        assert correlation.transaction_count("STORE_A") == 2
    finally:
        os.unlink(path)


def test_load_csv_deduplication(correlation):
    """Duplicate transaction_id in CSV must be loaded only once."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("store_id,transaction_id,timestamp,basket_value_inr\n")
        f.write(f"STORE_A,TXN_001,{make_ts(0)},100.00\n")
        f.write(f"STORE_A,TXN_001,{make_ts(0)},100.00\n")  # duplicate
        path = f.name
    try:
        loaded = correlation.load_csv(path)
        assert loaded == 1
    finally:
        os.unlink(path)


def test_load_csv_production_format(correlation):
    """Production Brigade Bangalore CSV with order_id / order_date / order_time columns."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(
            "order_id,coupon_code,offer_name,discount_code,invoice_number,invoice_type,"
            "order_date,order_time,return_id,store_id,store_name,city,customer_name,"
            "customer_number,sku,product_id,ean,product_name,brand_name,dep_name,"
            "sub_category,brand_type,tax,hsn_code,salesperson_id,employee_code,"
            "salesperson_name,qty,GMV,NMV,coupon_amount,item_promotion,amt_without_gwp,"
            "total_amount,pb_eb_sale,week_assigned,tax_m,taxable_amt,tax_amt\n"
        )
        f.write(
            "104363838,,Buy 2 Get 1 on PB,,ML0426KAP0001358,sales,10-04-2026,16:55:36,,"
            "ST1008,Brigade_Bangalore,Bangalore,Guest,9346413680,PPLBDD8904362534994NM2,"
            "402813,8.90436E+12,DERMDOC Body Wash,DERMDOC,bath-and-body,Body Wash,PB,18,"
            "33049990,1178,CL2063,kasthuri v,1,400,274.36,0,125.64,274.36,274.36,274.36,,1.18,232.51,41.85\n"
        )
        path = f.name
    try:
        loaded = correlation.load_csv(path)
        assert loaded == 1
        txns = correlation._transactions["ST1008"]
        assert txns[0].transaction_id == "104363838"
        assert txns[0].basket_value_inr == pytest.approx(274.36)
        assert txns[0].timestamp == "2026-04-10T16:55:36Z"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 2. add_transaction
# ---------------------------------------------------------------------------

def test_add_transaction(correlation):
    txn = POSTransaction(store_id="STORE_A", transaction_id="TXN_001", timestamp=make_ts(0))
    correlation.add_transaction(txn)
    assert correlation.transaction_count("STORE_A") == 1


# ---------------------------------------------------------------------------
# 3. correlate — within window
# ---------------------------------------------------------------------------

def test_correlate_within_window(correlation):
    session = make_session("VIS_01", queue_offset=0)
    # POS txn at 3 minutes (180 s) — within 5-min (300 s) window
    add_txn(correlation, "STORE_TEST", offset=180)
    count = correlation.correlate([session], "STORE_TEST")
    assert count == 1
    assert correlation.is_converted(session.session_id) is True


def test_correlate_outside_window(correlation):
    session = make_session("VIS_01", queue_offset=0)
    # POS txn at 6 minutes (360 s) — outside 5-min window
    add_txn(correlation, "STORE_TEST", offset=360)
    count = correlation.correlate([session], "STORE_TEST")
    assert count == 0
    assert correlation.is_converted(session.session_id) is False


def test_correlate_staff_session_skipped(correlation):
    session = make_session("STAFF_01", is_staff=True, queue_offset=0)
    add_txn(correlation, "STORE_TEST", offset=60)
    count = correlation.correlate([session], "STORE_TEST")
    assert count == 0


def test_correlate_no_billing_zone_skipped(correlation):
    session = make_session("VIS_01", has_billing_queue=False)
    add_txn(correlation, "STORE_TEST", offset=60)
    count = correlation.correlate([session], "STORE_TEST")
    assert count == 0


# ---------------------------------------------------------------------------
# 4. conversion_rate
# ---------------------------------------------------------------------------

def test_conversion_rate_zero(correlation):
    sessions = [make_session("VIS_01", has_billing_queue=False)]
    assert correlation.conversion_rate(sessions, "STORE_TEST") == 0.0


def test_conversion_rate_full(correlation):
    sessions = [make_session(f"VIS_{i}", queue_offset=0) for i in range(3)]
    add_txn(correlation, "STORE_TEST", offset=60)
    correlation.correlate(sessions, "STORE_TEST")
    rate = correlation.conversion_rate(sessions, "STORE_TEST")
    assert rate == 1.0


def test_conversion_rate_partial(correlation):
    s1 = make_session("VIS_01", queue_offset=0)
    s2 = make_session("VIS_02", has_billing_queue=False)
    add_txn(correlation, "STORE_TEST", offset=60)
    correlation.correlate([s1, s2], "STORE_TEST")
    rate = correlation.conversion_rate([s1, s2], "STORE_TEST")
    assert rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 5. Idempotency
# ---------------------------------------------------------------------------

def test_correlate_idempotent(correlation):
    session = make_session("VIS_01", queue_offset=0)
    add_txn(correlation, "STORE_TEST", offset=60)
    first_run = correlation.correlate([session], "STORE_TEST")
    second_run = correlation.correlate([session], "STORE_TEST")
    assert first_run == 1
    assert second_run == 0  # already converted, not re-converted


# ---------------------------------------------------------------------------
# 6. clear()
# ---------------------------------------------------------------------------

def test_correlation_clear(correlation):
    session = make_session("VIS_01", queue_offset=0)
    add_txn(correlation, "STORE_TEST", offset=60)
    correlation.correlate([session], "STORE_TEST")
    assert correlation.is_converted(session.session_id) is True

    correlation.clear()
    assert correlation.is_converted(session.session_id) is False
    assert correlation.transaction_count("STORE_TEST") == 0
