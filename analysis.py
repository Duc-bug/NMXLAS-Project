import cv2
import numpy as np
import matplotlib.pyplot as plt
from signature_recognition import SignatureRecognition
import os

def analyze_signature_features(image_path):
    """Phân tích chi tiết đặc trưng của một chữ ký"""
    
    recognizer = SignatureRecognition()
    
    # Đọc ảnh gốc
    original = cv2.imread(image_path)
    if original is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return
    
    # Tiền xử lý
    processed = recognizer.preprocess_image(image_path)
    if processed is None:
        print("Lỗi tiền xử lý ảnh")
        return
    
    # Trích xuất đặc trưng
    features = recognizer.extract_features(processed)
    if features is None:
        print("Lỗi trích xuất đặc trưng")
        return
    
    # Hiển thị kết quả
    plt.figure(figsize=(15, 10))
    
    # Ảnh gốc
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Ảnh gốc')
    plt.axis('off')
    
    # Ảnh đã xử lý
    plt.subplot(2, 3, 2)
    plt.imshow(processed, cmap='gray')
    plt.title('Ảnh đã xử lý')
    plt.axis('off')
    
    # Histogram của ảnh xử lý
    plt.subplot(2, 3, 3)
    plt.hist(processed.ravel(), bins=256, range=[0, 256])
    plt.title('Histogram')
    plt.xlabel('Giá trị pixel')
    plt.ylabel('Tần suất')
    
    # Contours
    plt.subplot(2, 3, 4)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(processed)
    cv2.drawContours(contour_img, contours, -1, 255, 2)
    plt.imshow(contour_img, cmap='gray')
    plt.title('Contours')
    plt.axis('off')
    
    # Thống kê đặc trưng
    plt.subplot(2, 3, 5)
    feature_stats = [
        f"Pixel features: {len(features)-10}",
        f"Mean intensity: {features[-10]:.2f}",
        f"Std intensity: {features[-9]:.2f}",
        f"Area: {features[-8]:.2f}",
        f"Perimeter: {features[-7]:.2f}",
        f"Area/Perimeter: {features[-6]:.2f}"
    ]
    
    plt.text(0.1, 0.9, '\n'.join(feature_stats), transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    plt.title('Đặc trưng trích xuất')
    plt.axis('off')
    
    # Hu Moments
    plt.subplot(2, 3, 6)
    hu_moments = features[-7:]
    plt.bar(range(len(hu_moments)), hu_moments)
    plt.title('Hu Moments')
    plt.xlabel('Moment index')
    plt.ylabel('Giá trị')
    
    plt.tight_layout()
    plt.show()
    
    return features

def compare_signatures(image1_path, image2_path):
    """So sánh hai chữ ký và hiển thị kết quả"""
    
    recognizer = SignatureRecognition()
    
    # Xử lý ảnh 1
    processed1 = recognizer.preprocess_image(image1_path)
    features1 = recognizer.extract_features(processed1)
    
    # Xử lý ảnh 2
    processed2 = recognizer.preprocess_image(image2_path)
    features2 = recognizer.extract_features(processed2)
    
    if features1 is None or features2 is None:
        print("Lỗi xử lý ảnh")
        return
    
    # Tính độ tương đồng
    from sklearn.metrics.pairwise import cosine_similarity
    
    min_len = min(len(features1), len(features2))
    features1_trimmed = features1[:min_len]
    features2_trimmed = features2[:min_len]
    
    similarity = cosine_similarity(
        features1_trimmed.reshape(1, -1),
        features2_trimmed.reshape(1, -1)
    )[0][0]
    
    # Hiển thị kết quả
    plt.figure(figsize=(15, 8))
    
    # Ảnh 1
    plt.subplot(2, 3, 1)
    original1 = cv2.imread(image1_path)
    plt.imshow(cv2.cvtColor(original1, cv2.COLOR_BGR2RGB))
    plt.title(f'Ảnh 1: {os.path.basename(image1_path)}')
    plt.axis('off')
    
    # Ảnh 2
    plt.subplot(2, 3, 2)
    original2 = cv2.imread(image2_path)
    plt.imshow(cv2.cvtColor(original2, cv2.COLOR_BGR2RGB))
    plt.title(f'Ảnh 2: {os.path.basename(image2_path)}')
    plt.axis('off')
    
    # Ảnh xử lý 1
    plt.subplot(2, 3, 4)
    plt.imshow(processed1, cmap='gray')
    plt.title('Ảnh 1 đã xử lý')
    plt.axis('off')
    
    # Ảnh xử lý 2
    plt.subplot(2, 3, 5)
    plt.imshow(processed2, cmap='gray')
    plt.title('Ảnh 2 đã xử lý')
    plt.axis('off')
    
    # Kết quả so sánh
    plt.subplot(2, 3, 3)
    colors = ['red' if similarity < 0.5 else 'orange' if similarity < 0.7 else 'green']
    plt.bar(['Độ tương đồng'], [similarity], color=colors)
    plt.ylim(0, 1)
    plt.title(f'Độ tương đồng: {similarity:.3f}')
    plt.ylabel('Cosine Similarity')
    
    # Biểu đồ so sánh đặc trưng
    plt.subplot(2, 3, 6)
    sample_features = min(50, min_len)  # Chỉ hiển thị 50 đặc trưng đầu
    x = range(sample_features)
    plt.plot(x, features1_trimmed[:sample_features], 'b-', label='Ảnh 1', alpha=0.7)
    plt.plot(x, features2_trimmed[:sample_features], 'r-', label='Ảnh 2', alpha=0.7)
    plt.title('So sánh đặc trưng')
    plt.xlabel('Feature index')
    plt.ylabel('Feature value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Kết luận
    if similarity >= 0.8:
        conclusion = "Rất có thể cùng một chữ ký"
    elif similarity >= 0.6:
        conclusion = "Có thể cùng một chữ ký"
    elif similarity >= 0.4:
        conclusion = "Không chắc chắn"
    else:
        conclusion = "Có thể khác chữ ký"
    
    print(f"Độ tương đồng: {similarity:.3f}")
    print(f"Kết luận: {conclusion}")
    
    return similarity

def batch_analysis(image_folder):
    """Phân tích hàng loạt các ảnh chữ ký trong thư mục"""
    
    if not os.path.exists(image_folder):
        print(f"Thư mục không tồn tại: {image_folder}")
        return
    
    recognizer = SignatureRecognition()
    
    # Lấy danh sách file ảnh
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(image_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_folder, file))
    
    if not image_files:
        print("Không tìm thấy file ảnh nào trong thư mục")
        return
    
    print(f"Tìm thấy {len(image_files)} file ảnh")
    print("Đang phân tích...")
    
    results = []
    
    for i, image_path in enumerate(image_files):
        print(f"Xử lý {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        processed = recognizer.preprocess_image(image_path)
        if processed is not None:
            features = recognizer.extract_features(processed)
            if features is not None:
                results.append({
                    'file': os.path.basename(image_path),
                    'path': image_path,
                    'features': features,
                    'processed': processed
                })
            else:
                print(f"  ✗ Lỗi trích xuất đặc trưng")
        else:
            print(f"  ✗ Lỗi tiền xử lý")
    
    # Tạo ma trận tương đồng
    if len(results) > 1:
        print("\nTạo ma trận tương đồng...")
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = np.zeros((len(results), len(results)))
        
        for i in range(len(results)):
            for j in range(len(results)):
                if i != j:
                    features1 = results[i]['features']
                    features2 = results[j]['features']
                    
                    min_len = min(len(features1), len(features2))
                    features1_trimmed = features1[:min_len]
                    features2_trimmed = features2[:min_len]
                    
                    similarity = cosine_similarity(
                        features1_trimmed.reshape(1, -1),
                        features2_trimmed.reshape(1, -1)
                    )[0][0]
                    
                    similarity_matrix[i][j] = similarity
                else:
                    similarity_matrix[i][j] = 1.0
        
        # Hiển thị ma trận tương đồng
        plt.figure(figsize=(12, 10))
        
        file_names = [result['file'] for result in results]
        
        plt.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Độ tương đồng')
        plt.title('Ma trận tương đồng giữa các chữ ký')
        
        plt.xticks(range(len(file_names)), file_names, rotation=45, ha='right')
        plt.yticks(range(len(file_names)), file_names)
        
        # Thêm giá trị vào từng ô
        for i in range(len(results)):
            for j in range(len(results)):
                if i != j:
                    text = plt.text(j, i, f'{similarity_matrix[i][j]:.2f}',
                                   ha="center", va="center", color="white", fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        # Tìm các cặp tương đồng cao
        print("\nCác cặp chữ ký có độ tương đồng cao (>= 0.7):")
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                if similarity_matrix[i][j] >= 0.7:
                    print(f"  {results[i]['file']} - {results[j]['file']}: {similarity_matrix[i][j]:.3f}")
    
    print(f"\nHoàn thành phân tích {len(results)} ảnh")

if __name__ == "__main__":
    # Tạo dữ liệu mẫu
    from demo import create_sample_signatures
    create_sample_signatures()
    
    print("=== PHÂN TÍCH CHI TIẾT CHỮ KÝ ===")
    print("\n1. Phân tích đặc trưng của một chữ ký:")
    analyze_signature_features("sample_signature_1.png")
    
    print("\n2. So sánh hai chữ ký:")
    compare_signatures("sample_signature_1.png", "sample_signature_2.png")
    
    print("\n3. Phân tích hàng loạt:")
    batch_analysis(".")
