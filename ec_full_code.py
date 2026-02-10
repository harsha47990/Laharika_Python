
from PIL import ImageFont
from PIL import Image, ImageDraw
import re
import os
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import numpy as np
import cv2  # OpenCV for image processing

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def non_max_suppression(cells, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove duplicate/overlapping cell detections.
    Keeps the larger bounding box when duplicates are found.
    """
    if len(cells) == 0:
        return cells
    
    # Sort by area (largest first)
    cells_sorted = sorted(cells, key=lambda c: c['area'], reverse=True)
    
    keep = []
    while len(cells_sorted) > 0:
        # Take the largest remaining cell
        current = cells_sorted.pop(0)
        keep.append(current)
        
        # Remove all cells that overlap significantly with current
        cells_sorted = [
            cell for cell in cells_sorted
            if calculate_iou(current['bbox'], cell['bbox']) < iou_threshold
        ]
    
    return keep

def filter_cells_by_aspect_ratio(cells, tolerance=0.20):
    """
    Remove cells with odd aspect ratios compared to the median.
    This helps remove merged cells or partial detections.
    
    Args:
        cells (list): List of detected cell dictionaries
        tolerance (float): Acceptable deviation from median (default: 0.25 = 25%)
        
    Returns:
        list: Filtered cells with consistent aspect ratios
    """
    if len(cells) == 0:
        return cells
    
    # Calculate aspect ratio for each cell
    for cell in cells:
        cell['aspect_ratio'] = cell['width'] / cell['height'] if cell['height'] > 0 else 0
    
    # Find median aspect ratio
    aspect_ratios = [c['aspect_ratio'] for c in cells]
    aspect_ratios.sort()
    median_ar = aspect_ratios[len(aspect_ratios) // 2]
    
    #print(f"  Median aspect ratio: {median_ar:.3f}")
    
    # Filter cells within tolerance
    min_ar = median_ar * (1 - tolerance)
    max_ar = median_ar * (1 + tolerance)
    
    filtered = [c for c in cells if min_ar <= c['aspect_ratio'] <= max_ar]
    
    removed = len(cells) - len(filtered)
    if removed > 0:
        print(f"  Removed {removed} cells with odd aspect ratios")
    
    return filtered

def hybrid_grid_fill(detected_cells, top_left, bottom_right, expected_rows=7, expected_cols=3):
    """
    Hybrid approach: Keep detected cells and fill missing grid positions.
    Maps detected cells to grid positions, then fills gaps with calculated rectangles.
    
    Args:
        detected_cells (list): List of detected cell dictionaries
        top_left (tuple): Grid top-left corner (left, top)
        bottom_right (tuple): Grid bottom-right corner (right, bottom)
        expected_rows (int): Number of rows
        expected_cols (int): Number of columns
        
    Returns:
        list: Complete grid with 21 cells (detected + calculated)
    """
    print(f"\n🔧 Applying Hybrid Grid Fill...")
    
    left, top = top_left
    right, bottom = bottom_right
    
    total_width = right - left
    total_height = bottom - top
    
    cell_width = total_width / expected_cols
    cell_height = total_height / expected_rows
    
    # Create grid template
    grid = {}  # Key: (row, col), Value: cell dict
    
    # Map detected cells to grid positions
    for cell in detected_cells:
        cx, cy = cell['center']
        
        # Calculate grid position
        col = int((cx - left) / cell_width)
        row = int((cy - top) / cell_height)
        
        # Clamp to valid range
        col = max(0, min(expected_cols - 1, col))
        row = max(0, min(expected_rows - 1, row))
        
        # Store cell in grid
        grid_key = (row, col)
        if grid_key not in grid:
            grid[grid_key] = cell
            grid[grid_key]['grid_position'] = grid_key
        else:
            # Position conflict - keep the one closest to grid center
            existing_cell = grid[grid_key]
            
            # Calculate expected center for this grid position
            expected_cx = left + col * cell_width + cell_width / 2
            expected_cy = top + row * cell_height + cell_height / 2
            
            # Distance from expected center
            dist_new = ((cx - expected_cx)**2 + (cy - expected_cy)**2)**0.5
            ex_cx, ex_cy = existing_cell['center']
            dist_existing = ((ex_cx - expected_cx)**2 + (ex_cy - expected_cy)**2)**0.5
            
            if dist_new < dist_existing:
                grid[grid_key] = cell
                grid[grid_key]['grid_position'] = grid_key
    
    #print(f"  ✓ Mapped {len(grid)} detected cells to grid positions")
    
    # Fill missing positions with calculated rectangles
    filled_count = 0
    for row in range(expected_rows):
        for col in range(expected_cols):
            grid_key = (row, col)
            
            if grid_key not in grid:
                # Calculate rectangle for this position
                x1 = left + col * cell_width
                y1 = top + row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                # Create calculated cell
                grid[grid_key] = {
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': (x1 + cell_width/2, y1 + cell_height/2),
                    'width': cell_width,
                    'height': cell_height,
                    'area': cell_width * cell_height,
                    'grid_position': grid_key,
                    'is_calculated': True  # Mark as calculated
                }
                filled_count += 1
    
    print(f"  ✓ Filled {filled_count} missing positions with calculated rectangles")
    
    # Convert grid to sorted list
    complete_cells = []
    for row in range(expected_rows):
        for col in range(expected_cols):
            cell = grid[(row, col)]
            complete_cells.append(cell)
    
    return complete_cells

def detect_cells_from_image(page_image, top_left, bottom_right, expected_rows=7, expected_cols=3, output_folder=None, page_num=None, total_pages=None):
    """
    Detect individual ID card boxes using contour detection.
    Each ID card has a bounding box/rectangle. Finds all 21 cells (7×3).
    Uses minimum area threshold of 85% of expected cell size.
    
    Args:
        page_image (PIL.Image): The page image
        top_left (tuple): Top-left corner of the grid (left, top)
        bottom_right (tuple): Bottom-right corner of the grid (right, bottom)
        expected_rows (int): Number of rows (default: 7)
        expected_cols (int): Number of columns (default: 3)
        output_folder (str): Optional output folder to save errored pages
        page_num (int): Optional page number for filename
        total_pages (int): Optional total number of pages for filename formatting
    Returns:
        list: List of dictionaries with 'bbox' (x1, y1, x2, y2) and 'center' (x, y) for each detected cell
    """
    #print(f"\nDetecting ID card boxes using contour detection...")
    
    # Convert PIL image to OpenCV format
    img_array = np.array(page_image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Calculate expected cell dimensions
    left, top = top_left
    right, bottom = bottom_right
    
    total_width = right - left
    total_height = bottom - top
    total_area = total_width * total_height
    
    expected_cell_count = expected_rows * expected_cols
    expected_cell_area = total_area / expected_cell_count
    
    # Minimum area threshold: 85% of expected cell size
    min_area = expected_cell_area * 0.80
    max_area = expected_cell_area * 1.2  # Max 120% to avoid merging
    
    # print(f"Total grid area: {total_area:,.0f} pixels²")
    # print(f"Expected cell area: {expected_cell_area:,.0f} pixels²")
    # print(f"Minimum cell area (85%): {min_area:,.0f} pixels²")
    # print(f"Maximum cell area (120%): {max_area:,.0f} pixels²")
    
    # Apply adaptive thresholding to find card boundaries
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to clean up and find rectangles
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_rect, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    #print(f"Found {len(contours)} total contours")
    
    # Filter contours by area
    detected_cells = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Filter by area (85% to 120% of expected cell size)
        if min_area <= area <= max_area:
            detected_cells.append({
                'bbox': (x, y, x + w, y + h),
                'center': (x + w/2, y + h/2),
                'width': w,
                'height': h,
                'area': area
            })
    
    #print(f"Filtered to {len(detected_cells)} cells by area threshold")
    
    # Apply Non-Maximum Suppression to remove duplicates
    temp_total_detected_cells = detected_cells.copy()
    if len(detected_cells) > 0:
        detected_cells = non_max_suppression(detected_cells, iou_threshold=0.5)
        print(f"After NMS: {len(detected_cells)} cells")
    
    # Filter out cells with odd aspect ratios
    if len(detected_cells) > 0:
        detected_cells = filter_cells_by_aspect_ratio(detected_cells, tolerance=0.25)
        print(f"After aspect ratio filter: {len(detected_cells)} cells")
    
    # If missing cells, apply hybrid grid fill
    if len(detected_cells) < expected_cell_count and len(detected_cells) > 0 and page_num != total_pages:
        print(f"⚠️  Only {len(detected_cells)} cells detected. Applying Hybrid Grid Fill...")
        detected_cells = hybrid_grid_fill(detected_cells, top_left, bottom_right, expected_rows, expected_cols)
        print(f"✓ After Hybrid Fill: {len(detected_cells)} cells")
    
    # Sort cells in reading order (top to bottom, left to right)
    expected_width = total_width / expected_cols
    expected_height = total_height / expected_rows
    detected_cells.sort(key=lambda c: (c['center'][1] // (expected_height * 0.8), c['center'][0]))
    
    print(f"✓ Detected {len(detected_cells)} cells (expected: {expected_cell_count})")
    
    # Visualize detected cells only if count is less than expected
    if len(detected_cells) < expected_cell_count:
        #print(f"⚠️  WARNING: Missing cells! Showing visualization...")

        # Number the cells
        for idx, cell in enumerate(detected_cells):
            cell['cell_number'] = idx + 1
        
        # Number the temp cells as well (for visualization)
        temp_total_detected_cells.sort(key=lambda c: (c['center'][1] // (expected_height * 0.8), c['center'][0]))
        for idx, cell in enumerate(temp_total_detected_cells):
            cell['cell_number'] = idx + 1

        # Create visualization
        vis_image = create_grid_visualization(page_image, detected_cells, top_left, bottom_right, page_num if page_num else 0)
        full_detected_img = create_grid_visualization(page_image, temp_total_detected_cells, top_left, bottom_right, page_num if page_num else 0)
        # Save to errored_pages folder if output_folder is provided
        if output_folder and page_num is not None:
            errored_folder = os.path.join(output_folder, "errored_pages")
            os.makedirs(errored_folder, exist_ok=True)
            
            error_filename = f"page_{page_num}.png"
            error_filepath = os.path.join(errored_folder, error_filename)
            vis_image.save(error_filepath)
            full_detected_filepath = os.path.join(errored_folder, f"page_{page_num}_full_detected.png")
            full_detected_img.save(full_detected_filepath)
            print(f"  💾 Saved errored page to: errored_pages/{error_filename}")
    
    return detected_cells

def process_pdf_grid(pdf_path, top_left, bottom_right, rows=7, cols=3, dpi=300, output_folder=None):
    """
    Process a PDF containing a grid of ID cards.
    Creates a fixed grid from the detected boundary and divides it into equal cells.
    Extracts text directly from PDF and assigns to cells.
    Saves each grid cell as an image. Memory efficient - processes one page at a time.
    
    Args:
        pdf_path (str): Path to the PDF file
        top_left (tuple): (left, top) coordinates of the grid boundary in IMAGE space
        bottom_right (tuple): (right, bottom) coordinates of the grid boundary in IMAGE space
        rows (int): Number of rows (default: 7)
        cols (int): Number of columns (default: 3)
        dpi (int): DPI used for image conversion (default: 300)
        output_folder (str): Folder to save grid cell images (optional)
        
    Returns:
        list: List of dictionaries containing position and extracted door numbers for all pages
    """
    try:
        print(f"\n{'='*60}")
        print(f"Processing PDF: {pdf_path}")
        print(f"Grid: {rows} rows x {cols} columns")
        print(f"Region (image coords): {top_left} to {bottom_right}")
        if output_folder:
            print(f"Output folder: {output_folder}")
        print(f"{'='*60}\n")
        
        # Create output folder if specified
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            print(f"Created output folder: {output_folder}")
        
        # Open PDF with PyMuPDF to get page count
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        if total_pages == 0:
            print("Error: No pages found in PDF")
            doc.close()
            return []
        
        print(f"Total pages in PDF: {total_pages}")
        
        # Calculate padding for page numbers
        page_num_width = len(str(total_pages))
        
        # Convert image coordinates to PDF coordinates
        scale_factor = 72 / dpi
        
        x1_img, y1_img = top_left
        x2_img, y2_img = bottom_right
        
        x1 = x1_img * scale_factor
        y1 = y1_img * scale_factor
        x2 = x2_img * scale_factor
        y2 = y2_img * scale_factor
        
        # Calculate cell dimensions
        total_width = x2 - x1
        total_height = y2 - y1
        cell_width = total_width / cols
        cell_height = total_height / rows
        
        # Image dimensions
        img_cell_width = (x2_img - x1_img) / cols
        img_cell_height = (y2_img - y1_img) / rows
        
        print(f"Cell dimensions (PDF coords): {cell_width:.2f} x {cell_height:.2f}")
        print(f"Cell dimensions (Image): {img_cell_width:.2f} x {img_cell_height:.2f}")
        
        # All results across all pages (only store door numbers, not images)
        all_results = []
        
        # Process each page ONE AT A TIME (memory efficient)
        for page_num in range(total_pages):
            print(f"\n{'='*60}")
            print(f"Processing Page {page_num + 1}/{total_pages}")
            print(f"{'='*60}")
            
            # Get the current page from PDF
            page = doc[page_num]
            
            # Convert ONLY this page to image (not all pages at once)
 
            page_images = convert_from_path(
                pdf_path, 
                dpi=dpi, 
                first_page=page_num + 1,
                last_page=page_num + 1,
                poppler_path=r"C:\Users\harsha.martha\Downloads\poppler-24.02.0\Library\bin"
            )
            page_image = page_images[0]
 
            
            # Create fixed grid cells from detected boundary
            detected_cells = detect_cells_from_image(page_image, top_left, bottom_right, rows, cols, output_folder, page_num + 1,total_pages)
                
            # Extract text from this page
            text_instances = page.get_text("dict")
            
            # Process each detected cell
            for cell_idx, cell in enumerate(detected_cells):
                img_x1, img_y1, img_x2, img_y2 = cell['bbox']
                
                # Convert image coordinates to PDF coordinates
                pdf_x1 = img_x1 * scale_factor
                pdf_y1 = img_y1 * scale_factor
                pdf_x2 = img_x2 * scale_factor
                pdf_y2 = img_y2 * scale_factor
                
                # Collect text lines within this cell
                cell_texts = []
                
                for block in text_instances["blocks"]:
                    if block["type"] == 0:  # Text block
                        for line in block["lines"]:
                            line_bbox = line["bbox"]
                            line_x = (line_bbox[0] + line_bbox[2]) / 2
                            line_y = (line_bbox[1] + line_bbox[3]) / 2
                            
                            # Extract text from line
                            line_text = ""
                            for span in line["spans"]:
                                line_text += span["text"]
                            
                            line_text = line_text.strip()
                            if not line_text:
                                continue
                            
                            # Check if line is within this cell's bounding box
                            if pdf_x1 <= line_x <= pdf_x2 and pdf_y1 <= line_y <= pdf_y2:
                                cell_texts.append({
                                    'text': line_text,
                                    'y': line_y,
                                    'x': line_x
                                })
                
                # Extract door numbers
                door_numbers = extract_door_from_cell_texts(cell_texts, pdf_y2)
                
                # Save cell image if output folder specified
                if output_folder:
                    # Crop the cell from page image
                    cell_image = page_image.crop((
                        int(img_x1), 
                        int(img_y1),
                        int(img_x2), 
                        int(img_y2)
                    ))
                    
                    # Generate filename: page_001_cell_01 format
                    page_str = str(page_num + 1).zfill(page_num_width)
                    cell_str = str(cell_idx + 1).zfill(2)
                    filename = f"page_{page_str}_cell_{cell_str}.png"
                    if door_numbers:
                        door_str = door_numbers[0].replace("/", "_").replace(" ", "").replace("-", "_")
                        filename = f"{door_str}.png"
                 
                    filepath = os.path.join(output_folder, filename)
                    i = 1
                    while os.path.exists(filepath):
                        filename = f"{door_str}_{i}.png"
                        filepath = os.path.join(output_folder, filename)
                        i += 1
                    cell_image.save(filepath)
                    print(f"  💾 Saved: {filename}")
                
                # Store only door numbers and metadata (not images)
                result = {
                    'page': page_num + 1,
                    'cell': cell_idx + 1,
                    'position': f"[Page {page_num + 1}, Cell {cell_idx + 1}]",
                    'bbox_image': (int(img_x1), int(img_y1), int(img_x2), int(img_y2)),
                    'door_numbers': door_numbers,
                    'total_lines': len(cell_texts),
                    'image_file': f"page_{str(page_num + 1).zfill(page_num_width)}_cell_{str(cell_idx + 1).zfill(2)}.png" if output_folder else None
                }
                all_results.append(result)
                
            
            # Clear page image from memory after processing this page
            del page_image
            del page_images
            del detected_cells
            
            print(f"✓ Page {page_num + 1} processing complete, memory released")
        
        doc.close()
        print(f"\n{'='*60}")
        print(f"Completed processing all {total_pages} pages")
        print(f"Total cells processed: {len(all_results)}")
        print(f"{'='*60}")
        
        return all_results
        
    except Exception as e:
        print(f"Error processing PDF grid: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def extract_door_from_cell_texts(texts, cell_bottom_y):
    """
    Extract door numbers from text lines in a cell.
    Looks for lines matching door number patterns.
    
    Args:
        texts (list): List of dictionaries with 'text', 'y', 'x' keys
        cell_bottom_y (float): Bottom Y coordinate of the cell
        
    Returns:
        list: List of door numbers found in the cell
    """
    if not texts:
        print("  No text found in this cell")
        return []
    
    # Sort texts by Y position (top to bottom)
    sorted_texts = sorted(texts, key=lambda t: t['y'])
    
    # Find all lines that match door number pattern
    door_numbers = []
    door_pattern = re.compile(r':?\d+[-/]\d+(?:[-/]\d+)?(?:/\d+)?')
    ## below commented will match all patterns below
    # 
    # 3-8-6 ✓
    # 3 - 8 - 6 ✓
    # 3 7 16 ✓
    # 3/8/6 ✓
    # 3-5-82/1/A ✓
    ############
    #door_pattern = re.compile(r':?\d+[\s\-/]+\d+(?:[\s\-/]+\d+)?(?:/\d+)?(?:/[A-Za-z0-9]+)?')

    for text_item in sorted_texts:
        #tried but failed text = text_item['text'].strip().replace(" ",'')
        text = text_item['text'].strip()

        # Check if this line matches door number pattern
        match = door_pattern.search(text)
        if match:
            door_num = match.group(0)
            # Remove leading colon if present
            if door_num.startswith(':'):
                door_num = door_num[1:]
            
            # Skip if it looks like an ID or serial number (too many parts)
            parts = re.split(r'[-/]', door_num)
            if len(parts) <= 4:  # Door numbers typically have 2-4 parts
                door_numbers.append(door_num)
                #print(f"  ✓ Found door number: '{door_num}' in line: '{text}'")
    
    if not door_numbers:
        print("  ✗ No door numbers found")
       
    return door_numbers

def detect_grid_boundary(pdf_path, page_number=1, dpi=300, show_preview=True):
    """
    Automatically detect the bounding box of the grid on a PDF page.
    Finds the largest rectangle which should be the grid boundary.
    
    Args:
        pdf_path (str): Path to the PDF file
        page_number (int): Page number to analyze (default: 1)
        dpi (int): DPI for conversion (default: 300)
        show_preview (bool): Show visual preview of detected boundary (default: True)
        
    Returns:
        tuple: (top, left, bottom, right) coordinates of the detected grid boundary
    """
    print(f"\n{'='*60}")
    print(f"Detecting grid boundary on page {page_number}")
    print(f"PDF: {pdf_path}")
    print(f"{'='*60}\n")
    
    # Convert the specified page to image
    print(f"Converting page {page_number} to image at {dpi} DPI...")
    page_images = convert_from_path(
        pdf_path, 
        first_page=page_number,
        last_page=page_number,
        dpi=dpi,
        poppler_path=r"C:\Users\harsha.martha\Downloads\poppler-24.02.0\Library\bin"
    )
    
    if not page_images:
        print("Error: Could not convert PDF page to image")
        return None
    
    page_image = page_images[0]
    print(f"Page converted: {page_image.size[0]}x{page_image.size[1]} pixels")
    
    # Convert PIL image to OpenCV format
    img_array = np.array(page_image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    print("\nDetecting grid boundary using edge detection...")
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to connect nearby contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Error: No contours detected")
        return None
    
    # Find the largest contour (should be the grid boundary)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Coordinates of the detected box
    left = x
    top = y
    right = x + w
    bottom = y + h
    
    print(f"\n✓ Detected grid boundary:")
    print(f"  Top-Left: ({left}, {top})")
    print(f"  Bottom-Right: ({right}, {bottom})")
    print(f"  Width: {right - left} pixels")
    print(f"  Height: {bottom - top} pixels")
    print(f"  Area: {(right - left) * (bottom - top):,} pixels²")
    
    # Show preview if requested
    if show_preview:
        preview_image = page_image.copy()
        draw = ImageDraw.Draw(preview_image)
        
        # Draw the detected rectangle with green color
        draw.rectangle([left, top, right, bottom], outline='green', width=8)
        
        # Draw crosshairs at corners
        marker_size = 30
        # Top-left corner
        draw.line([(left - marker_size, top), (left + marker_size, top)], fill='blue', width=4)
        draw.line([(left, top - marker_size), (left, top + marker_size)], fill='blue', width=4)
        # Bottom-right corner
        draw.line([(right - marker_size, bottom), (right + marker_size, bottom)], fill='blue', width=4)
        draw.line([(right, bottom - marker_size), (right, bottom + marker_size)], fill='blue', width=4)
        
        try:
            font = ImageFont.truetype("arial.ttf", 50)
        except:
            font = ImageFont.load_default()
        
        # Draw coordinate labels
        draw.text((left + 10, top + 10), f"({left}, {top})", fill='blue', font=font)
        draw.text((right - 300, bottom - 80), f"({right}, {bottom})", fill='blue', font=font)
        
        # Display dimensions
        width_px = right - left
        height_px = bottom - top
        draw.text((left + width_px//2 - 150, top - 80), f"Width: {width_px}px", fill='green', font=font)
        draw.text((left - 300, top + height_px//2), f"Height: {height_px}px", fill='green', font=font)
        
        print(f"\nShowing preview of detected boundary...")
        preview_image.show()
    
    return (top, left, bottom, right)

def create_grid_visualization(page_image, detected_cells, top_left, bottom_right, page_num):
    """
    Create a visualization of detected cells without showing it.
    Returns the annotated image.
    
    Args:
        page_image (PIL.Image): The page image
        detected_cells (list): List of detected cell dictionaries
        top_left (tuple): Grid top-left corner
        bottom_right (tuple): Grid bottom-right corner
        page_num (int): Page number for labeling
        
    Returns:
        PIL.Image: Annotated image with grid visualization
    """
    # Create a copy for drawing
    vis_image = page_image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    # Extract grid boundaries
    left, top = top_left
    right, bottom = bottom_right
    
    # Draw outer boundary (green)
    draw.rectangle([left, top, right, bottom], outline='green', width=5)
    
    try:
        font = ImageFont.truetype("arial.ttf", 25)
        font_large = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
        font_large = ImageFont.load_default()
    
    # Draw each detected cell
    for cell in detected_cells:
        x1, y1, x2, y2 = cell['bbox']
        cell_num = cell['cell_number']
        
        # Draw cell boundary (red)
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        
        # Draw cell number in center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        text = str(cell_num)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw background circle
        circle_radius = 20
        draw.ellipse(
            [center_x - circle_radius, center_y - circle_radius,
             center_x + circle_radius, center_y + circle_radius],
            fill='yellow', outline='black', width=2
        )
        
        # Draw cell number
        draw.text(
            (center_x - text_width/2, center_y - text_height/2),
            text,
            fill='black',
            font=font
        )
    
    # Add info at top
    info_text = f"Page {page_num}: {len(detected_cells)} cells"
    draw.text((left + 10, top - 80), info_text, fill='blue', font=font_large)
    
    return vis_image


import os
from PIL import Image, ImageDraw, ImageFont
import re
MM_TO_INCH = 1 / 25.4

def windows_natural_sort_key(s):
    """
    Sort key with hierarchical grouping by numeric parts.
    Groups files like: 3_8_935, 3_8_935_1, 3_8_935_2, then 3_8_936
    
    Splits by underscore, converts numbers to integers, keeps separators.
    This ensures proper hierarchical grouping.
    """
    # Extract just the filename without extension
    if '.' in s:
        name, ext = s.rsplit('.', 1)
    else:
        name = s
        ext = ''
    
    # Split by underscore and convert numeric parts to integers
    parts = name.split('_')
    key = []
    for part in parts:
        if part.isdigit():
            # Use tuple (0, int_value) for numbers - 0 sorts before strings
            key.append((0, int(part)))
        else:
            # Use tuple (1, string_value) for strings - 1 sorts after numbers
            key.append((1, part.lower()))
    
    # Add extension at the end for secondary sorting
    key.append((1, ext.lower()))
    
    return key

def images_to_a4_pages(
    image_folder,
    output_folder,
    rows=7,
    cols=3,
    dpi=300,
    margin_top_mm=15,
    margin_bottom_mm=15,
    margin_left_mm=20,
    margin_right_mm=10,
    orientation='auto',  # 'auto', 'portrait', or 'landscape'
    preserve_aspect_ratio=True,  # True: maintain aspect ratio, False: allow stretching
    max_stretch_percent=25,  # Maximum stretch allowed (percentage)
    min_row_gap_mm=2  # Minimum gap between rows in mm
):
    """
    Reads images from a folder and creates A4-sized image pages
    laid out in rows x cols format.
    
    Args:
        orientation: 'auto' (landscape if cols>rows, else portrait), 'portrait', or 'landscape'
        preserve_aspect_ratio: If True, maintain original aspect ratio. If False, allow stretching.
        max_stretch_percent: Maximum percentage to stretch images (only when preserve_aspect_ratio=False)
        min_row_gap_mm: Minimum gap between rows in millimeters
    """

    os.makedirs(output_folder, exist_ok=True)

    # Determine orientation
    if orientation == 'auto':
        is_landscape = cols >= rows
    elif orientation == 'landscape':
        is_landscape = True
    else:
        is_landscape = False

    # A4 size in pixels
    if is_landscape:
        a4_width_px = int(11.69 * dpi)
        a4_height_px = int(8.27 * dpi)
        print(f"Using LANDSCAPE orientation (cols={cols} > rows={rows})")
    else:
        a4_width_px = int(8.27 * dpi)
        a4_height_px = int(11.69 * dpi)
        print(f"Using PORTRAIT orientation")

    def mm_to_px(mm):
        return int(mm * MM_TO_INCH * dpi)

    # Convert layout measurements
    margin_top = mm_to_px(margin_top_mm)
    margin_bottom = mm_to_px(margin_bottom_mm)
    margin_left = mm_to_px(margin_left_mm)
    margin_right = mm_to_px(margin_right_mm)
    min_row_gap = mm_to_px(min_row_gap_mm)

    # Read images in computer-sort order
    images = sorted(
        (
            f for f in os.listdir(image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
        ),
        key=windows_natural_sort_key
    )
    
    if not images:
        raise ValueError("No images found in the input folder.")

    usable_width = a4_width_px - margin_left - margin_right
    usable_height = a4_height_px - margin_top - margin_bottom

    # Calculate cell dimensions with row gaps
    total_row_gaps = (rows - 1) * min_row_gap
    available_height = usable_height - total_row_gaps
    
    cell_width = usable_width // cols
    cell_height = available_height // rows

    print(f"Cell dimensions: {cell_width}x{cell_height} pixels")
    print(f"Preserve aspect ratio: {preserve_aspect_ratio}")
    if not preserve_aspect_ratio:
        print(f"Max stretch allowed: {max_stretch_percent}%")
    print(f"Row gap: {min_row_gap_mm}mm ({min_row_gap}px)")

    images_per_page = rows * cols
    page_index = 1
    img_index = 0

    while img_index < len(images):
        page = Image.new("RGB", (a4_width_px, a4_height_px), "white")

        for r in range(rows):
            for c in range(cols):
                if img_index >= len(images):
                    break

                img_path = os.path.join(image_folder, images[img_index])
                img = Image.open(img_path).convert("RGB")
                
                # Calculate cell position with row gaps
                cell_x = margin_left + c * cell_width
                cell_y = margin_top + r * (cell_height + min_row_gap)

                if preserve_aspect_ratio:
                    # Maintain aspect ratio - fit within cell
                    img.thumbnail((cell_width, cell_height), Image.LANCZOS)
                    
                    # Center image within the cell
                    x = cell_x + (cell_width - img.width) // 2
                    y = cell_y + (cell_height - img.height) // 2
                    
                    page.paste(img, (x, y))
                else:
                    # Allow stretching up to max_stretch_percent
                    orig_width, orig_height = img.size
                    orig_aspect = orig_width / orig_height
                    cell_aspect = cell_width / cell_height
                    
                    # Calculate initial scaled size maintaining aspect ratio
                    if orig_aspect > cell_aspect:
                        # Image is wider - will have gaps on top/bottom
                        # Scale based on width first
                        new_width = cell_width
                        new_height = int(cell_width / orig_aspect)
                        
                        # Now stretch height to fill gap (up to max allowed)
                        height_gap = cell_height - new_height
                        max_allowed_stretch = new_height * (max_stretch_percent / 100)
                        
                        if height_gap > 0:
                            # Stretch by the minimum of: gap size or max allowed stretch
                            actual_stretch = min(height_gap, max_allowed_stretch)
                            new_height = int(new_height + actual_stretch)
                    else:
                        # Image is taller - will have gaps on left/right
                        # Scale based on height first
                        new_height = cell_height
                        new_width = int(cell_height * orig_aspect)
                        
                        # Now stretch width to fill gap (up to max allowed)
                        width_gap = cell_width - new_width
                        max_allowed_stretch = new_width * (max_stretch_percent / 100)
                        
                        if width_gap > 0:
                            # Stretch by the minimum of: gap size or max allowed stretch
                            actual_stretch = min(width_gap, max_allowed_stretch)
                            new_width = int(new_width + actual_stretch)
                    
                    # Resize image (with stretching applied)
                    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Center image within the cell
                    x = cell_x + (cell_width - new_width) // 2
                    y = cell_y + (cell_height - new_height) // 2
                    
                    page.paste(img_resized, (x, y))

                img_index += 1

        # Add page number at bottom right corner
        draw = ImageDraw.Draw(page)
        page_text = f"Page No: {page_index}"
        
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), page_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position at bottom right with margin
        text_x = a4_width_px - text_width - mm_to_px(10)
        text_y = a4_height_px - text_height - mm_to_px(10)
        
        draw.text((text_x, text_y), page_text, fill='black', font=font)

        output_path = os.path.join(output_folder, f"page__{page_index:04d}.png")
        page.save(output_path, dpi=(dpi, dpi))
        print(f"Saved: page__{page_index:04d}.png")
        page_index += 1

def a4_images_to_a3_pdf_simple(
    input_folder,
    output_pdf,
    dpi=300
):
    """
    Uses Pillow only.
    Combines A4 images into A3 spreads and saves as a multi-page PDF.
    Auto-detects portrait/landscape A4 input and adjusts A3 layout accordingly.
    """

    # Read images in computer sort order
    file_names = sorted(
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    )

    if not file_names:
        raise ValueError("No images found in input folder.")

    # Detect orientation from first image
    first_img = Image.open(os.path.join(input_folder, file_names[0]))
    is_landscape_input = first_img.width > first_img.height
    
    if is_landscape_input:
        # Input: Landscape A4 (11.69" × 8.27")
        # Output: Portrait A3 (11.69" × 16.54") - stack vertically
        a3_width_px = int(11.69 * dpi)
        a3_height_px = int(16.54 * dpi)
        print(f"Detected LANDSCAPE A4 input → Creating PORTRAIT A3 (stacking vertically)")
    else:
        # Input: Portrait A4 (8.27" × 11.69")
        # Output: Landscape A3 (16.54" × 11.69") - side by side
        a3_width_px = int(16.54 * dpi)
        a3_height_px = int(11.69 * dpi)
        print(f"Detected PORTRAIT A4 input → Creating LANDSCAPE A3 (side by side)")

    # If odd, duplicate last page
    if len(file_names) % 2 != 0:
        file_names.append(file_names[-1])

    half = len(file_names) // 2
    first_half = file_names[:half]
    second_half = file_names[half:]

    a3_pages = []

    for left_name, right_name in zip(first_half, second_half):
        left_img = Image.open(os.path.join(input_folder, left_name)).convert("RGB")
        right_img = Image.open(os.path.join(input_folder, right_name)).convert("RGB")

        page = Image.new("RGB", (a3_width_px, a3_height_px), "white")

        if is_landscape_input:
            # Stack vertically for landscape input
            half_height = a3_height_px // 2
            
            # Resize to fit half A3 height
            left_img.thumbnail((a3_width_px, half_height), Image.LANCZOS)
            right_img.thumbnail((a3_width_px, half_height), Image.LANCZOS)
            
            # Center horizontally
            left_x = (a3_width_px - left_img.width) // 2
            right_x = (a3_width_px - right_img.width) // 2
            
            page.paste(left_img, (left_x, 0))
            page.paste(right_img, (right_x, half_height))
        else:
            # Side by side for portrait input
            half_width = a3_width_px // 2
            
            # Resize to fit half A3 width
            left_img.thumbnail((half_width, a3_height_px), Image.LANCZOS)
            right_img.thumbnail((half_width, a3_height_px), Image.LANCZOS)
            
            # Center vertically
            left_y = (a3_height_px - left_img.height) // 2
            right_y = (a3_height_px - right_img.height) // 2
            
            page.paste(left_img, (0, left_y))
            page.paste(right_img, (half_width, right_y))

        a3_pages.append(page)
        print(f"Created A3 page from: {left_name} and {right_name}")
        
    # Save as multi-page PDF
    a3_pages[0].save(
        output_pdf,
        save_all=True,
        append_images=a3_pages[1:],
        resolution=dpi
    )

def images_to_a3_pdf_direct(
    image_folder,
    output_pdf,
    rows=5,
    cols=3,
    dpi=300,
    margin_top_mm=10,
    margin_bottom_mm=10,
    margin_left_mm=20,
    margin_right_mm=10,
    preserve_aspect_ratio=False,
    max_stretch_percent=40,
    min_row_gap_mm=2
):
    """
    Memory-efficient: Creates A3 PDF directly from images without saving intermediate A4 files.
    Creates batches, splits them in half, pairs them up, and streams to A3 PDF.
    """
    print(f"\nCreating A3 PDF directly from images (memory-efficient mode)...")
    print(f"Input folder: {image_folder}")
    print(f"Output PDF: {output_pdf}")
    
    # Read and sort images
    images = sorted(
        (
            f for f in os.listdir(image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
        ),
        key=windows_natural_sort_key
    )
    
    if not images:
        raise ValueError("No images found in the input folder.")
    
    images_per_page = rows * cols
    
    # Create batches (each batch = one A4 page worth of images)
    batches = []
    for i in range(0, len(images), images_per_page):
        batch = images[i:i + images_per_page]
        batches.append(batch)
    
    print(f"Total images: {len(images)}")
    print(f"Images per page: {images_per_page}")
    print(f"Total batches: {len(batches)}")
    
    # If odd number of batches, duplicate the last batch
    if len(batches) % 2 != 0:
        batches.append(batches[-1])
        print(f"Odd number of batches, duplicated last batch. New total: {len(batches)}")
    
    # Split batches into two halves
    half_point = len(batches) // 2
    first_half = batches[:half_point]
    second_half = batches[half_point:]
    
    print(f"Split into two halves: {len(first_half)} + {len(second_half)} batches")
    
    # A4 and A3 dimensions
    is_landscape = cols >= rows
    
    if is_landscape:
        a4_width_px = int(11.69 * dpi)
        a4_height_px = int(8.27 * dpi)
        # A3 Portrait (stacking vertically)
        a3_width_px = int(11.69 * dpi)
        a3_height_px = int(16.54 * dpi)
        print(f"Using LANDSCAPE A4 → PORTRAIT A3 (stacking vertically)")
    else:
        a4_width_px = int(8.27 * dpi)
        a4_height_px = int(11.69 * dpi)
        # A3 Landscape (side by side)
        a3_width_px = int(16.54 * dpi)
        a3_height_px = int(11.69 * dpi)
        print(f"Using PORTRAIT A4 → LANDSCAPE A3 (side by side)")
    
    def mm_to_px(mm):
        return int((mm / 25.4) * dpi)
    
    # Calculate cell dimensions for A4 pages
    margin_top = mm_to_px(margin_top_mm)
    margin_bottom = mm_to_px(margin_bottom_mm)
    margin_left = mm_to_px(margin_left_mm)
    margin_right = mm_to_px(margin_right_mm)
    min_row_gap = mm_to_px(min_row_gap_mm)
    
    usable_width = a4_width_px - margin_left - margin_right
    usable_height = a4_height_px - margin_top - margin_bottom
    
    total_row_gaps = (rows - 1) * min_row_gap
    available_height = usable_height - total_row_gaps
    
    cell_width = usable_width // cols
    cell_height = available_height // rows
    
    def create_a4_page_from_batch(batch, page_num):
        """Create a single A4 page from a batch of images."""
        page = Image.new("RGB", (a4_width_px, a4_height_px), "white")
        
        for idx, img_filename in enumerate(batch):
            img_path = os.path.join(image_folder, img_filename)
            img = Image.open(img_path).convert("RGB")
            
            row = idx // cols
            col = idx % cols
            
            # Calculate cell position with row gaps
            cell_x = margin_left + col * cell_width
            cell_y = margin_top + row * (cell_height + min_row_gap)
            
            if preserve_aspect_ratio:
                img.thumbnail((cell_width, cell_height), Image.LANCZOS)
                x = cell_x + (cell_width - img.width) // 2
                y = cell_y + (cell_height - img.height) // 2
                page.paste(img, (x, y))
            else:
                # Stretching logic
                orig_width, orig_height = img.size
                orig_aspect = orig_width / orig_height
                cell_aspect = cell_width / cell_height
                
                if orig_aspect > cell_aspect:
                    new_width = cell_width
                    new_height = int(cell_width / orig_aspect)
                    height_gap = cell_height - new_height
                    max_allowed_stretch = new_height * (max_stretch_percent / 100)
                    if height_gap > 0:
                        actual_stretch = min(height_gap, max_allowed_stretch)
                        new_height = int(new_height + actual_stretch)
                else:
                    new_height = cell_height
                    new_width = int(cell_height * orig_aspect)
                    width_gap = cell_width - new_width
                    max_allowed_stretch = new_width * (max_stretch_percent / 100)
                    if width_gap > 0:
                        actual_stretch = min(width_gap, max_allowed_stretch)
                        new_width = int(new_width + actual_stretch)
                
                img_resized = img.resize((new_width, new_height), Image.LANCZOS)
                x = cell_x + (cell_width - new_width) // 2
                y = cell_y + (cell_height - new_height) // 2
                page.paste(img_resized, (x, y))
        
        # Add page number
        draw = ImageDraw.Draw(page)
        page_text = f"Page No: {page_num}"
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), page_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = a4_width_px - text_width - mm_to_px(10)
        text_y = a4_height_px - text_height - mm_to_px(10)
        draw.text((text_x, text_y), page_text, fill='black', font=font)
        
        return page
    
    # Create A3 pages by pairing batches from first and second halves
    a3_pages = []
    
    for idx, (batch1, batch2) in enumerate(zip(first_half, second_half)):
        print(f"Creating A3 page {idx + 1}/{len(first_half)}...")
        
        # Create two A4 pages with correct sequential page numbers
        # First half page numbers: 1, 2, 3, ...
        # Second half page numbers: (half_point + 1), (half_point + 2), ...
        page_num1 = idx + 1
        page_num2 = half_point + idx + 1
        
        a4_page1 = create_a4_page_from_batch(batch1, page_num1)
        a4_page2 = create_a4_page_from_batch(batch2, page_num2)
        
        # Create A3 page
        a3_page = Image.new("RGB", (a3_width_px, a3_height_px), "white")
        
        if is_landscape:
            # Stack vertically
            half_height = a3_height_px // 2
            a4_page1.thumbnail((a3_width_px, half_height), Image.LANCZOS)
            a4_page2.thumbnail((a3_width_px, half_height), Image.LANCZOS)
            x1 = (a3_width_px - a4_page1.width) // 2
            x2 = (a3_width_px - a4_page2.width) // 2
            a3_page.paste(a4_page1, (x1, 0))
            a3_page.paste(a4_page2, (x2, half_height))
        else:
            # Side by side
            half_width = a3_width_px // 2
            a4_page1.thumbnail((half_width, a3_height_px), Image.LANCZOS)
            a4_page2.thumbnail((half_width, a3_height_px), Image.LANCZOS)
            y1 = (a3_height_px - a4_page1.height) // 2
            y2 = (a3_height_px - a4_page2.height) // 2
            a3_page.paste(a4_page1, (0, y1))
            a3_page.paste(a4_page2, (half_width, y2))
        
        a3_pages.append(a3_page)
        
        # Free memory
        del a4_page1
        del a4_page2
    
    # Save as multi-page PDF
    print(f"Saving A3 PDF with {len(a3_pages)} pages...")
    a3_pages[0].save(
        output_pdf,
        save_all=True,
        append_images=a3_pages[1:],
        resolution=dpi
    )
    print(f"✓ Saved: {output_pdf}")

def process_pdf_creation(extracted_images_output_folder, a4_output_folder, a3_output_folder, create_a4_files=False):
    """
    Process PDF creation with option to bypass A4 file creation.
    
    Args:
        extracted_images_output_folder: Folder containing extracted images
        a4_output_folder: Folder to save A4 files (only used if create_a4_files=True)
        a3_output_folder: Folder to save A3 PDFs
        create_a4_files: If True, creates intermediate A4 files. If False, streams directly to A3 (memory-efficient)
    """
    
    for ward in os.listdir(extracted_images_output_folder):
        extracted_images_ward_folder = os.path.join(extracted_images_output_folder, ward)
        a3_ward_output_pdf = os.path.join(a3_output_folder, f"{ward}.pdf")
        
        if os.path.exists(a3_ward_output_pdf):
            print(f"A3 PDF already exists for ward {ward}, skipping...")
            continue
        
        if create_a4_files:
            # Old method: Create A4 files first, then merge to A3
            print(f"\nProcessing {ward} with A4 file creation...")
            a4_ward_output_folder = os.path.join(a4_output_folder, ward)
            
            if os.path.exists(a4_ward_output_folder):
                print(f"A4 files already exists for ward {ward}, skipping A4 creation...")
            else:
                images_to_a4_pages(
                    image_folder=extracted_images_ward_folder,
                    output_folder=a4_ward_output_folder,
                    margin_top_mm=10,
                    margin_bottom_mm=10,
                    margin_left_mm=20,
                    margin_right_mm=10,
                    preserve_aspect_ratio=False, 
                    max_stretch_percent=40,  
                    min_row_gap_mm=2,  
                    rows=5, cols=3
                )
            
            a4_images_to_a3_pdf_simple(
                input_folder=a4_ward_output_folder,
                output_pdf=a3_ward_output_pdf,
                dpi=300
            )
        else:
            # New method: Stream directly to A3 PDF (memory-efficient)
            print(f"\nProcessing {ward} in memory-efficient mode (no A4 files)...")
            images_to_a3_pdf_direct(
                image_folder=extracted_images_ward_folder,
                output_pdf=a3_ward_output_pdf,
                rows=3,
                cols=3,
                dpi=300,
                margin_top_mm=10,
                margin_bottom_mm=10,
                margin_left_mm=20,
                margin_right_mm=10,
                preserve_aspect_ratio=False,
                max_stretch_percent=40,
                min_row_gap_mm=2
            )
        
        print(f"Completed processing for ward {ward}.")
        
def process_voter_seperation(cleaned_pdf_folder, extracted_images_output_folder):

    for file in os.listdir(cleaned_pdf_folder):
        if file.lower().endswith(".pdf"):

            pdf_path = os.path.join(cleaned_pdf_folder, file)
            
            # Skip if already processed
            extracted_ward_output_folder = os.path.join(extracted_images_output_folder, os.path.splitext(file)[0])
            if os.path.exists(extracted_ward_output_folder):
                print(f"Skipping already processed file: {file}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing: {file}")
            print(f"{'='*60}")
            
            # Automatically detect grid boundary from first page
            detected_coords = detect_grid_boundary(pdf_path, page_number=3, dpi=300, show_preview=False)
         
            top, left, bottom, right = detected_coords
            top_left = (left, top)
            bottom_right = (right, bottom)
         
            # Create output folder and extract
            os.makedirs(extracted_ward_output_folder, exist_ok=True)
            
            # Use detected coordinates for extraction
            results = process_pdf_grid(pdf_path, top_left, bottom_right, rows=7, cols=3, output_folder=extracted_ward_output_folder)
            
            print("\n" + "="*60 + "\n")
            print(f"✓ Completed extraction for: {file}")
            print(f"  Total cells extracted: {len(results)}")
            

def main():
    main_folder = r"C:\Users\harsha.martha\Downloads\EC2"
    cleaned_pdf_folder = os.path.join(main_folder, "raw pdf")
    extracted_images_output_folder = os.path.join(main_folder, "extracted_imgs")
    a4_output_folder = os.path.join(main_folder, "a4")
    a3_output_folder = os.path.join(main_folder, "output pdf")
    for folder in [main_folder, cleaned_pdf_folder, extracted_images_output_folder, a4_output_folder, a3_output_folder]:
        os.makedirs(folder, exist_ok=True)  

    process_voter_seperation(cleaned_pdf_folder, extracted_images_output_folder)
    # process_pdf_creation(extracted_images_output_folder, a4_output_folder, a3_output_folder,
    #                      create_a4_files=False)

if __name__ == "__main__":
    main()
    






###########################################################


# import os
# from PIL import Image

# # =========================
# # Sorting (as provided)
# # =========================

# def windows_natural_sort_key(s):
#     if '.' in s:
#         name, ext = s.rsplit('.', 1)
#     else:
#         name = s
#         ext = ''

#     parts = name.split('_')
#     key = []
#     for part in parts:
#         if part.isdigit():
#             key.append((0, int(part)))
#         else:
#             key.append((1, part.lower()))

#     key.append((1, ext.lower()))
#     return key


# # =========================
# # Page / Layout Constants
# # =========================

# DPI = 300

# # A3 portrait @ 300 DPI
# A3_WIDTH_PX  = 3508
# A3_HEIGHT_PX = 4961

# COLS = 3
# ROWS = 5
# IMAGES_PER_PAGE = 15

# # Stretch height by X percent (e.g. 10 = +10%)
# HEIGHT_STRETCH_PERCENT = 30


# # =========================
# # Margin Helpers
# # =========================

# def mm_to_px(mm, dpi=DPI):
#     return int((mm / 25.4) * dpi)

# MARGIN_TOP_PX    = mm_to_px(15)
# MARGIN_BOTTOM_PX = mm_to_px(15)
# MARGIN_LEFT_PX   = mm_to_px(20)
# MARGIN_RIGHT_PX  = mm_to_px(10)


# # =========================
# # Resize + Stretch Logic
# # =========================

# def resize_and_stretch_height(img, cell_width, cell_height, stretch_percent):
#     """
#     1. Resize to fit cell while preserving aspect ratio
#     2. Stretch height by stretch_percent
#     3. Clamp height to cell_height
#     """

#     img_ratio = img.width / img.height
#     cell_ratio = cell_width / cell_height

#     # Step 1: aspect-ratio fit
#     if img_ratio > cell_ratio:
#         new_width = cell_width
#         new_height = int(cell_width / img_ratio)
#     else:
#         new_height = cell_height
#         new_width = int(cell_height * img_ratio)

#     img = img.resize((new_width, new_height), Image.LANCZOS)

#     # Step 2: stretch height
#     stretch_factor = 1 + (stretch_percent / 100.0)
#     stretched_height = int(img.height * stretch_factor)

#     # Step 3: clamp to cell height
#     stretched_height = min(stretched_height, cell_height)

#     img = img.resize((img.width, stretched_height), Image.LANCZOS)

#     return img


# # =========================
# # PDF Generator
# # =========================

# def create_a3_pdf(image_dir, output_pdf):
#     image_files = [
#         f for f in os.listdir(image_dir)
#         if f.lower().endswith((".jpg", ".jpeg", ".png"))
#     ]

#     if not image_files:
#         raise ValueError("No image files found.")

#     image_files.sort(key=windows_natural_sort_key)
#     image_files = [os.path.join(image_dir, f) for f in image_files]

#     printable_width = A3_WIDTH_PX - MARGIN_LEFT_PX - MARGIN_RIGHT_PX
#     printable_height = A3_HEIGHT_PX - MARGIN_TOP_PX - MARGIN_BOTTOM_PX

#     cell_width = printable_width // COLS
#     cell_height = printable_height // ROWS

#     pages = []

#     for i in range(0, len(image_files), IMAGES_PER_PAGE):
#         page = Image.new("RGB", (A3_WIDTH_PX, A3_HEIGHT_PX), "white")
#         batch = image_files[i:i + IMAGES_PER_PAGE]

#         for idx, img_path in enumerate(batch):
#             img = Image.open(img_path).convert("RGB")

#             img = resize_and_stretch_height(
#                 img,
#                 cell_width,
#                 cell_height,
#                 HEIGHT_STRETCH_PERCENT
#             )

#             col = idx % COLS
#             row = idx // COLS

#             x = (
#                 MARGIN_LEFT_PX
#                 + col * cell_width
#                 + (cell_width - img.width) // 2
#             )

#             y = (
#                 MARGIN_TOP_PX
#                 + row * cell_height
#                 + (cell_height - img.height) // 2
#             )

#             page.paste(img, (x, y))

#         pages.append(page)

#     pages[0].save(
#         output_pdf,
#         save_all=True,
#         append_images=pages[1:],
#         resolution=DPI
#     )


# # =========================
# # Example Usage
# # =========================

# if __name__ == "__main__":
#     create_a3_pdf(
#         image_dir=r"C:\Users\harsha.martha\Downloads\EC\extracted_imgs\ward_33",
#         output_pdf=r"C:\Users\harsha.martha\Downloads\EC\output pdf\ward_33_A3_15_images_per_page.pdf"
#     )
