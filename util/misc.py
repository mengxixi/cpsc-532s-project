from PIL import Image, ImageDraw, ImageFont


def inference_image(image_fname, gt_boxes, gt_prop_boxes, pred_boxes, phrase):
    image = Image.open(image_fname).convert('RGB')
    draw = ImageDraw.Draw(image)

    # Draw the proposals that I'm supposed to predict
    # Draw before predicted boxes, so if we predicted correctly, this won't
    # show up
    for box in gt_prop_boxes:
        drawrect(draw, box, outline='yellow', width=3)

    # Draw predicted boxes
    for box in pred_boxes:
        drawrect(draw, box, outline='red', width=3)

    # Draw gt boxes
    for box in gt_boxes:
        drawrect(draw, box, outline='green', width=3)


    # Write text
    strip_h = 25
    bg = Image.new('RGB', (image.size[0], image.size[1]+2*strip_h), color='black')
    drawbg = ImageDraw.Draw(bg)
    font = ImageFont.truetype('/home/siyi/fonts/Calibri.ttf', 15)
    drawbg.text((0,0), ' '+image_fname.split('/')[-1][:-4], fill='white', font=font)
    drawbg.text((0,strip_h+image.size[1]), ' '+phrase, fill='white', font=font)
    bg.paste(image, (0,strip_h))
    return bg


# https://stackoverflow.com/questions/34255938/is-there-a-way-to-specify-the-width-of-a-rectangle-in-pil
def drawrect(drawcontext, xy, outline=None, width=1):
    x1 = xy[0]; y1 = xy[1]; x2=xy[2]; y2=xy[3]
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)