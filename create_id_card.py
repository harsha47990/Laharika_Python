from PIL import Image, ImageDraw, ImageFont


def split_text(S, font, width):
    words = S.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        text_width = font.getbbox(test_line)[2]  # The width is at index 2

        
        if text_width <= width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines
   
def add_address_to_image(image,text,position, margin, font_path, font_size, next_line_buffer):
    # Open the image
    draw = ImageDraw.Draw(image)
    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Get image width
    image_width = int(image.width)
    x, y = position
    text_width = image_width - x - margin
    
    # Wrap text
    lines = split_text(text,font,text_width)
   
    # Calculate line height
    _, line_height = draw.textbbox((0, 0), "A", font=font)[2:]
    
    # Draw text on image
    y = position[1]
    for line in lines:
        line_width, _ = draw.textbbox((0, 0), line, font=font)[2:]
        draw.text((position[0], y), line, font=font, fill=(0, 0, 0))  
        y += line_height + next_line_buffer

    return y

def add_fields_to_image(image, fields, position, font_path, font_size, next_line_buffer, draw_colon = False, collon_buffer = 25): 
    draw = ImageDraw.Draw(image)
    
    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Get image width
    image_width = image.width
    
    # Calculate the maximum width of the field names
    max_field_width = max([draw.textbbox((0, 0), field, font=font)[2] for field in fields])
    
    # Define the X position for the colons (after the longest field)
    colon_x = position[0] + max_field_width + collon_buffer  # Adding a little margin for space after the field name
    
    y = position[1]
    _, line_height = draw.textbbox((0, 0), "A", font=font)[2:]

    for field in fields:
        # Draw the field name
        draw.text((position[0], y), field, font=font, fill=(0, 0, 0))  # Black text
        
        if draw_colon:
            # Draw the colon at the fixed position
            draw.text((colon_x, y), ":", font=font, fill=(0, 0, 0))
        
        # Adjust y for the next line
        y += line_height + next_line_buffer  # Line spacing (height of the font + some space)
       
    return colon_x, y

def create_front_page(template_path,output_path, start_position, fields, values, fields_font="arialbd.ttf",value_font="arial.ttf",colon_value_buffer = 30, field_font_size = 30, value_font_size = 30, next_line_buffer = 10 ):
    
    image = Image.open(template_path)
    if image.mode == "RGBA":
        image = image.convert("RGB")

    colon_x_position, y = add_fields_to_image(image, fields, start_position, draw_colon = True, font_path=fields_font, font_size=field_font_size,next_line_buffer = next_line_buffer)

    colon_x_position += colon_value_buffer

    add_fields_to_image(image, values, (colon_x_position, start_position[1]), draw_colon = False, font_path=value_font, font_size=value_font_size, next_line_buffer = next_line_buffer)

    image.save(output_path, format="JPEG")

def create_back_page(template_path,output_path, start_position, fields, values, fields_font="arialbd.ttf",value_font="arial.ttf",colon_value_buffer = 30, field_font_size = 30, value_font_size = 30, next_line_buffer = 30):
    
    image = Image.open(template_path)
    if image.mode == "RGBA":
        image = image.convert("RGB")

    colon_x_position, _ = add_fields_to_image(image, fields, start_position, draw_colon = True, font_path=fields_font, font_size=field_font_size,next_line_buffer = next_line_buffer)

    colon_x_position += colon_value_buffer

    _, y = add_fields_to_image(image, values[:-1], (colon_x_position, start_position[1]), draw_colon = False, font_path=value_font, font_size=value_font_size, next_line_buffer = next_line_buffer)
    
    add_address_to_image(image,values[-1],(colon_x_position, y), margin=20,  font_path=value_font, font_size=value_font_size, next_line_buffer = next_line_buffer)
    
    image.save(output_path, format="JPEG")



front_template = r"C:\Users\harsha.martha\OneDrive - WIRB-Copernicus Group, Inc\Documents\harsha\id\front.png"
front_start_position = (40, 550)


fields = ["Name", "W/o - S/o", "Age", "Employee ID", "VO Name", "Village", "Mandal", "Dist"]
values = ["Martha Harsha", "Martha Ramesh", "26","15115245", "KMR","KRR","DFDFDF","VFDfDFDf"]
create_front_page(front_template, "output1.jpg", front_start_position,fields,values)


back_template = r"C:\Users\harsha.martha\OneDrive - WIRB-Copernicus Group, Inc\Documents\harsha\id\bck.png"
back_start_position = (40, 200)

fields = ["Card Validity", "Blood Group", "Cell No", "Address"]
values = ["Martha Harsha", "Martha Ramesh", "9948775758","5-5-200, vivekananda colony, kamareddy, 503111."]
create_back_page(back_template, "output2.jpg", back_start_position,fields,values,field_font_size = 35, value_font_size = 35, next_line_buffer = 40)
