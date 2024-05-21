import cv2
from pyzbar.pyzbar import decode

def read_barcode(image):
    barcodes = decode(image)
    if barcodes:
        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")
            barcode_rect = barcode.rect
            return barcode_data, barcode_rect
    return None, None

