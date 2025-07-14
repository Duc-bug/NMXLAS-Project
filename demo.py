import cv2
import numpy as np
from signature_recognition import SignatureRecognition
import matplotlib.pyplot as plt

def demo_signature_recognition():
    """Demo các chức năng chính của phần mềm nhận dạng chữ ký"""
    
    # Tạo đối tượng nhận dạng chữ ký
    recognizer = SignatureRecognition()
    
    print("=== DEMO PHẦN MỀM NHẬN DẠNG CHỮ KÝ ===")
    print()
    
    # Tạo ảnh chữ ký mẫu (demo)
    create_sample_signatures()
    
    # Demo thêm chữ ký vào cơ sở dữ liệu
    print("1. Thêm chữ ký vào cơ sở dữ liệu:")
    signatures = [
        ("John Doe", "sample_signature_1.png"),
        ("Jane Smith", "sample_signature_2.png"),
        ("Bob Johnson", "sample_signature_3.png")
    ]
    
    for name, file_path in signatures:
        success, message = recognizer.add_signature(name, file_path)
        print(f"  - {name}: {message}")
    
    print()
    
    # Demo liệt kê chữ ký
    print("2. Danh sách chữ ký đã lưu:")
    saved_signatures = recognizer.list_signatures()
    for i, sig in enumerate(saved_signatures, 1):
        print(f"  {i}. {sig}")
    
    print()
    
    # Demo nhận dạng chữ ký
    print("3. Nhận dạng chữ ký:")
    test_images = [
        "sample_signature_1.png",
        "sample_signature_2.png", 
        "sample_signature_3.png"
    ]
    
    for test_image in test_images:
        print(f"  Nhận dạng {test_image}:")
        result, message = recognizer.recognize_signature(test_image)
        if result:
            print(f"    ✓ Kết quả: {result} ({message})")
        else:
            print(f"    ✗ {message}")
    
    print()
    print("=== DEMO HOÀN THÀNH ===")

def create_sample_signatures():
    """Tạo các ảnh chữ ký mẫu để demo"""
    
    # Tạo ảnh chữ ký 1
    img1 = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img1, "John Doe", (50, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.line(img1, (50, 120), (300, 120), (0, 0, 0), 2)
    cv2.imwrite("sample_signature_1.png", img1)
    
    # Tạo ảnh chữ ký 2
    img2 = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img2, "Jane Smith", (50, 100), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, (0, 0, 0), 2)
    cv2.ellipse(img2, (200, 130), (100, 20), 0, 0, 180, (0, 0, 0), 2)
    cv2.imwrite("sample_signature_2.png", img2)
    
    # Tạo ảnh chữ ký 3
    img3 = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img3, "B. Johnson", (50, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.8, (0, 0, 0), 2)
    # Vẽ một số đường trang trí
    cv2.line(img3, (50, 120), (250, 120), (0, 0, 0), 1)
    cv2.line(img3, (50, 125), (200, 125), (0, 0, 0), 1)
    cv2.imwrite("sample_signature_3.png", img3)
    
    print("Đã tạo các ảnh chữ ký mẫu:")
    print("  - sample_signature_1.png")
    print("  - sample_signature_2.png")
    print("  - sample_signature_3.png")
    print()

def test_preprocessing():
    """Test chức năng tiền xử lý ảnh"""
    print("=== TEST TIỀN XỬ LÝ ẢNH ===")
    
    recognizer = SignatureRecognition()
    
    # Tạo ảnh test
    create_sample_signatures()
    
    # Test tiền xử lý
    for i in range(1, 4):
        img_path = f"sample_signature_{i}.png"
        print(f"Tiền xử lý {img_path}...")
        
        processed = recognizer.preprocess_image(img_path)
        if processed is not None:
            # Lưu ảnh đã xử lý
            output_path = f"processed_signature_{i}.png"
            cv2.imwrite(output_path, processed)
            print(f"  ✓ Đã lưu ảnh xử lý: {output_path}")
        else:
            print(f"  ✗ Lỗi xử lý ảnh")
    
    print()

if __name__ == "__main__":
    # Chạy demo
    demo_signature_recognition()
    
    # Test tiền xử lý
    test_preprocessing()
