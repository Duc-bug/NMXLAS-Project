import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import os
import pickle
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

class SignatureRecognition:
    def __init__(self):
        self.signatures_db = {}
        self.scaler = StandardScaler()
        self.model_path = "signature_model.pkl"
        self.load_model()
    
    def preprocess_image(self, image_path):
        """Tiền xử lý ảnh chữ ký"""
        try:
            # Đọc ảnh
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Không thể đọc ảnh từ {image_path}")
            
            # Chuyển sang ảnh xám
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Áp dụng bộ lọc Gaussian để giảm nhiễu
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Chuyển sang ảnh nhị phân (đen trắng)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Tìm contours để xác định vùng chữ ký
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Tìm contour lớn nhất (giả sử là chữ ký)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Cắt vùng chữ ký
                signature_region = binary[y:y+h, x:x+w]
                
                # Resize về kích thước chuẩn
                signature_resized = cv2.resize(signature_region, (200, 100))
                
                return signature_resized
            else:
                # Nếu không tìm thấy contours, resize toàn bộ ảnh
                return cv2.resize(binary, (200, 100))
                
        except Exception as e:
            print(f"Lỗi khi tiền xử lý ảnh: {str(e)}")
            return None
    
    def extract_features(self, processed_image):
        """Trích xuất đặc trưng từ ảnh đã xử lý"""
        if processed_image is None:
            return None
        
        features = []
        
        # 1. Đặc trưng pixel (flatten image)
        pixel_features = processed_image.flatten()
        features.extend(pixel_features)
        
        # 2. Đặc trưng thống kê
        mean_intensity = np.mean(processed_image)
        std_intensity = np.std(processed_image)
        features.extend([mean_intensity, std_intensity])
        
        # 3. Đặc trưng hình học
        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Tỷ lệ diện tích/chu vi
            if perimeter > 0:
                area_perimeter_ratio = area / perimeter
            else:
                area_perimeter_ratio = 0
            
            features.extend([area, perimeter, area_perimeter_ratio])
        else:
            features.extend([0, 0, 0])
        
        # 4. Đặc trưng Hu moments
        moments = cv2.moments(processed_image)
        hu_moments = cv2.HuMoments(moments)
        features.extend(hu_moments.flatten())
        
        return np.array(features)
    
    def add_signature(self, name, image_path):
        """Thêm chữ ký mới vào cơ sở dữ liệu"""
        processed_img = self.preprocess_image(image_path)
        if processed_img is None:
            return False, "Không thể xử lý ảnh"
        
        features = self.extract_features(processed_img)
        if features is None:
            return False, "Không thể trích xuất đặc trưng"
        
        self.signatures_db[name] = features
        self.save_model()
        return True, "Đã thêm chữ ký thành công"
    
    def recognize_signature(self, image_path, threshold=0.8):
        """Nhận dạng chữ ký"""
        if not self.signatures_db:
            return None, "Cơ sở dữ liệu chữ ký trống"
        
        processed_img = self.preprocess_image(image_path)
        if processed_img is None:
            return None, "Không thể xử lý ảnh"
        
        features = self.extract_features(processed_img)
        if features is None:
            return None, "Không thể trích xuất đặc trưng"
        
        best_match = None
        best_similarity = 0
        
        for name, stored_features in self.signatures_db.items():
            # Đảm bảo cùng kích thước features
            min_len = min(len(features), len(stored_features))
            features_trimmed = features[:min_len]
            stored_features_trimmed = stored_features[:min_len]
            
            # Tính toán độ tương đồng cosine
            similarity = cosine_similarity(
                features_trimmed.reshape(1, -1), 
                stored_features_trimmed.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        if best_similarity >= threshold:
            return best_match, f"Độ tương đồng: {best_similarity:.2f}"
        else:
            return None, f"Không tìm thấy chữ ký phù hợp (độ tương đồng cao nhất: {best_similarity:.2f})"
    
    def save_model(self):
        """Lưu mô hình"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.signatures_db, f)
        except Exception as e:
            print(f"Lỗi khi lưu mô hình: {str(e)}")
    
    def load_model(self):
        """Tải mô hình"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.signatures_db = pickle.load(f)
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {str(e)}")
            self.signatures_db = {}
    
    def list_signatures(self):
        """Liệt kê tất cả chữ ký trong cơ sở dữ liệu"""
        return list(self.signatures_db.keys())
    
    def remove_signature(self, name):
        """Xóa chữ ký khỏi cơ sở dữ liệu"""
        if name in self.signatures_db:
            del self.signatures_db[name]
            self.save_model()
            return True
        return False


class SignatureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Phần mềm nhận dạng chữ ký")
        self.root.geometry("800x600")
        
        self.signature_recognizer = SignatureRecognition()
        
        self.create_widgets()
    
    def create_widgets(self):
        # Tạo notebook (tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Thêm chữ ký
        self.add_tab = ttk.Frame(notebook)
        notebook.add(self.add_tab, text="Thêm chữ ký")
        self.create_add_tab()
        
        # Tab 2: Nhận dạng chữ ký
        self.recognize_tab = ttk.Frame(notebook)
        notebook.add(self.recognize_tab, text="Nhận dạng chữ ký")
        self.create_recognize_tab()
        
        # Tab 3: Quản lý chữ ký
        self.manage_tab = ttk.Frame(notebook)
        notebook.add(self.manage_tab, text="Quản lý chữ ký")
        self.create_manage_tab()
    
    def create_add_tab(self):
        # Frame cho thêm chữ ký
        main_frame = ttk.Frame(self.add_tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Tên chữ ký
        ttk.Label(main_frame, text="Tên chữ ký:").pack(anchor='w')
        self.name_entry = ttk.Entry(main_frame, width=50)
        self.name_entry.pack(pady=5)
        
        # Chọn file ảnh
        ttk.Label(main_frame, text="Chọn ảnh chữ ký:").pack(anchor='w', pady=(10, 0))
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill='x', pady=5)
        
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=60).pack(side='left', fill='x', expand=True)
        ttk.Button(file_frame, text="Chọn file", command=self.select_file_add).pack(side='right', padx=(5, 0))
        
        # Nút thêm chữ ký
        ttk.Button(main_frame, text="Thêm chữ ký", command=self.add_signature).pack(pady=20)
        
        # Kết quả
        self.add_result_text = ScrolledText(main_frame, height=10, width=70)
        self.add_result_text.pack(fill='both', expand=True, pady=10)
    
    def create_recognize_tab(self):
        # Frame cho nhận dạng chữ ký
        main_frame = ttk.Frame(self.recognize_tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Chọn file ảnh
        ttk.Label(main_frame, text="Chọn ảnh chữ ký cần nhận dạng:").pack(anchor='w')
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill='x', pady=5)
        
        self.recognize_file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.recognize_file_var, width=60).pack(side='left', fill='x', expand=True)
        ttk.Button(file_frame, text="Chọn file", command=self.select_file_recognize).pack(side='right', padx=(5, 0))
        
        # Ngưỡng độ tương đồng
        ttk.Label(main_frame, text="Ngưỡng độ tương đồng:").pack(anchor='w', pady=(10, 0))
        self.threshold_var = tk.DoubleVar(value=0.8)
        threshold_frame = ttk.Frame(main_frame)
        threshold_frame.pack(fill='x', pady=5)
        
        ttk.Scale(threshold_frame, from_=0.5, to=1.0, variable=self.threshold_var, orient='horizontal').pack(side='left', fill='x', expand=True)
        ttk.Label(threshold_frame, textvariable=self.threshold_var).pack(side='right', padx=(5, 0))
        
        # Nút nhận dạng
        ttk.Button(main_frame, text="Nhận dạng chữ ký", command=self.recognize_signature).pack(pady=20)
        
        # Kết quả
        self.recognize_result_text = ScrolledText(main_frame, height=10, width=70)
        self.recognize_result_text.pack(fill='both', expand=True, pady=10)
    
    def create_manage_tab(self):
        # Frame cho quản lý chữ ký
        main_frame = ttk.Frame(self.manage_tab)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Danh sách chữ ký
        ttk.Label(main_frame, text="Danh sách chữ ký đã lưu:").pack(anchor='w')
        
        # Listbox với scrollbar
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill='both', expand=True, pady=10)
        
        self.signature_listbox = tk.Listbox(list_frame)
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.signature_listbox.yview)
        self.signature_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.signature_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Nút điều khiển
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=10)
        
        ttk.Button(button_frame, text="Làm mới danh sách", command=self.refresh_signature_list).pack(side='left', padx=(0, 5))
        ttk.Button(button_frame, text="Xóa chữ ký", command=self.delete_signature).pack(side='left', padx=5)
        
        # Cập nhật danh sách ban đầu
        self.refresh_signature_list()
    
    def select_file_add(self):
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh chữ ký",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff")]
        )
        if file_path:
            self.file_path_var.set(file_path)
    
    def select_file_recognize(self):
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh chữ ký cần nhận dạng",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff")]
        )
        if file_path:
            self.recognize_file_var.set(file_path)
    
    def add_signature(self):
        name = self.name_entry.get().strip()
        file_path = self.file_path_var.get().strip()
        
        if not name:
            messagebox.showerror("Lỗi", "Vui lòng nhập tên chữ ký!")
            return
        
        if not file_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn file ảnh!")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Lỗi", "File không tồn tại!")
            return
        
        self.add_result_text.insert('end', f"Đang xử lý chữ ký '{name}'...\n")
        self.root.update()
        
        success, message = self.signature_recognizer.add_signature(name, file_path)
        
        if success:
            self.add_result_text.insert('end', f"✓ {message}\n")
            self.name_entry.delete(0, 'end')
            self.file_path_var.set('')
            self.refresh_signature_list()
        else:
            self.add_result_text.insert('end', f"✗ {message}\n")
        
        self.add_result_text.see('end')
    
    def recognize_signature(self):
        file_path = self.recognize_file_var.get().strip()
        threshold = self.threshold_var.get()
        
        if not file_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn file ảnh!")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Lỗi", "File không tồn tại!")
            return
        
        self.recognize_result_text.insert('end', f"Đang nhận dạng chữ ký...\n")
        self.root.update()
        
        result, message = self.signature_recognizer.recognize_signature(file_path, threshold)
        
        if result:
            self.recognize_result_text.insert('end', f"✓ Nhận dạng thành công: {result}\n")
            self.recognize_result_text.insert('end', f"  {message}\n")
        else:
            self.recognize_result_text.insert('end', f"✗ Không nhận dạng được\n")
            self.recognize_result_text.insert('end', f"  {message}\n")
        
        self.recognize_result_text.see('end')
    
    def refresh_signature_list(self):
        self.signature_listbox.delete(0, 'end')
        signatures = self.signature_recognizer.list_signatures()
        for signature in signatures:
            self.signature_listbox.insert('end', signature)
    
    def delete_signature(self):
        selection = self.signature_listbox.curselection()
        if not selection:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn chữ ký cần xóa!")
            return
        
        signature_name = self.signature_listbox.get(selection[0])
        
        if messagebox.askyesno("Xác nhận", f"Bạn có chắc chắn muốn xóa chữ ký '{signature_name}'?"):
            if self.signature_recognizer.remove_signature(signature_name):
                messagebox.showinfo("Thành công", f"Đã xóa chữ ký '{signature_name}'")
                self.refresh_signature_list()
            else:
                messagebox.showerror("Lỗi", "Không thể xóa chữ ký!")


if __name__ == "__main__":
    root = tk.Tk()
    app = SignatureGUI(root)
    root.mainloop()
