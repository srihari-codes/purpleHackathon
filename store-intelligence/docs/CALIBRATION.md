# Store Intelligence — Zone Calibration Guide

The Store Intelligence platform features a **fully dynamic zone configuration and mapping architecture**. All cameras, region shapes, and crossing lines are loaded at runtime. There are no hardcoded coordinates or camera layouts, and any changes saved in the Calibration Studio take effect immediately through the pipeline's live hot-reload engine without restarting the containers.

---

## Accessing the Zone Calibration Studio

The Calibration Studio can be accessed in two ways:
1. **Via the Onboarding Wizard (Recommended):** During Step 3 of the setup wizard (`http://localhost:8080`), click the **Calibrate** button next to any uploaded camera card. This will open the Calibration Studio for that specific camera slot.
2. **Direct Access:** Navigate to `http://localhost:8081` in your web browser. If no camera is active, select the camera from the left sidebar.

---

## Calibration Controls & Drawing Modes

The top navigation bar of the studio provides three primary modes for editing:

### 1. Select Mode (`↖ Select`)
- Use this mode to interact with existing shapes.
- Click inside any zone on the canvas to select it.
- **Modify Vertices:** Click and drag the white dots on the corners of any selected shape to reposition individual points.
- **Move Shape:** Click and drag the interior of a selected polygon to translate the entire shape.
- **Delete:** Select a shape and click the **🗑 Delete** button in the top bar or press the `Delete` key.

### 2. Draw Mode (`✏ Draw`)
- Used for drawing **polygons** representing floor zones, billing counters, queue regions, or staff areas.
- Click the canvas to place vertices.
- Click to place subsequent vertices.
- **Close the Shape:** Double-click or press the `Enter` key to connect the final point to the start point.

### 3. Entry Line Mode (`╱ Entry Line`)
- Used specifically for drawing **lines** representing the threshold crossing boundary.
- Click to place the first point (e.g., left side of entrance).
- Click to place the second point (e.g., right side of entrance).
- Once the second point is clicked, the shape is immediately committed as a line.

> [!IMPORTANT]
> **CRITICAL RULE FOR DRAWING ENTRY LINES:**
> While drawing an entry line, you **must only** use the dedicated **Entry Line** drawing option (`╱ Entry Line` button in the top bar near Select and Draw).
> **DO NOT** draw a polygon using Draw Mode and then attempt to change its role to `entry_line` in the properties sidebar.
> The line-crossing algorithm in `entry_exit.py` strictly expects a line segment defined by exactly two endpoints. Creating a polygon and changing its role to `entry_line` will result in mathematical validation failures or pipeline ingestion errors.

---

## Zone Roles & Property Configuration

When a shape is selected, its attributes can be edited in the **Properties** tab in the right sidebar:

| Role Name | Shape Type | Behavioral Effect on Detection Pipeline |
|---|---|---|
| `zone` | Polygon | General customer region. Tracks visits and accumulates dwell time (emits `ZONE_ENTER`, `ZONE_EXIT`, `ZONE_DWELL`). |
| `entry_line` | Line | Enters/exits threshold line. Tracks line-crossing vectors to emit `ENTRY` or `EXIT` events. |
| `inside_region` / `outside_region` | Polygon | Helper polygons defining which side of the entry line is inside vs. outside the store. |
| `billing_counter` / `queue_area` | Polygon | The billing queue boundary. Non-staff visitors occupying this shape are counted towards **Queue Depth** metrics and trigger `BILLING_QUEUE_JOIN` and `BILLING_QUEUE_ABANDON` events. |
| `staff_area` | Polygon | Bounding box crops whose foot-centroids fall inside this shape are unconditionally classified as staff (`is_staff: true`). |

---

## Step-by-Step Calibration Walkthrough

### Part A: Calibrating the Entry/Exit Camera (`CAM_ENTRY_01`)
1. Select the entry camera in the left sidebar.
2. Click **╱ Entry Line** in the top menu bar.
3. Click on the left side of the entry threshold, then click on the right side of the threshold.
4. Select the newly drawn line, open the **Properties** tab on the right, and:
   - Verify the role is set to `entry_line`.
   - Set the Shape ID to `ENTRY_LINE` and the Label to `ENTRY_DOOR`.
   - Set the **Inside direction** option (`below` or `above` the line).
5. (Optional but recommended) Click **✏ Draw** and draw a polygon on the inside of the door, and set its role to `inside_region`.

### Part B: Calibrating the Billing Counter (`CAM_BILLING_01`)
1. Select the billing camera in the left sidebar.
2. Click **＋ Add Zone** to create a shape, then click **✏ Draw**.
3. Draw a polygon bounding the queue layout area.
4. Select the shape, go to **Properties**, and:
   - Change the role to `queue_area` (or `billing_counter`).
   - Change the Shape ID to `ZONE_BILLING_QUEUE` or `QUEUE_AREA`.
   - Set a recognizable label (e.g., `Billing Queue Area`).
5. (Optional) Draw a small polygon directly behind the cash register, select it, and set the role to `staff_area`. This ensures cashiers standing behind the counter are automatically excluded from customer queue counts.

### Part C: Calibrating Floor browsing Zones
1. Select any floor camera (e.g., `CAM_FLOOR_01`).
2. Click **＋ Add Zone** and click **✏ Draw** to outline a specific rack or product category shelf (e.g., Skincare, Fragrances).
3. Under **Properties**:
   - Set the role to `zone`.
   - Set the Shape ID (e.g., `ZONE_SKINCARE`).
   - Set a label (e.g., `Skincare Section`).

---

## Saving and Live Hot-Reloading

1. Click the green **💾 Save** button in the topbar.
2. The calibration configuration is written directly to the host volume at `/data/calibration/{STORE_ID}.json`.
3. The detection pipeline's `ZoneMapper` monitors this directory. It detects the file modification and updates the active shapes **without restarting the camera streams**.
4. The live dashboard visualization immediately updates its overlays to reflect the new geometry and labels.
