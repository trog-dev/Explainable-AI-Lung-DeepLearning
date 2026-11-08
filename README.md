# Explainable AI cho phân loại viêm phổi từ ảnh X-quang ngực

Đồ án sử dụng **EfficientNetB0** và **ResNet152V2** để phân loại ảnh X-quang ngực thành hai lớp `NORMAL` và `PNEUMONIA`. Hai notebook áp dụng **Grad-CAM** và **LIME** để trực quan hóa các vùng ảnh ảnh hưởng đến dự đoán.

Đây là đồ án cơ sở ngành Khoa học máy tính tại Trường Đại học Sư phạm Kỹ thuật Vĩnh Long, năm 2025. Phạm vi của dự án là thực nghiệm phân loại ảnh và giải thích dự đoán.

## Mô hình và notebook

| Notebook | Mô hình | Thiết lập trong mã |
| --- | --- | --- |
| [efficientnetb0.ipynb](./efficientnetb0.ipynb) | EfficientNetB0 với đầu phân loại softmax hai lớp | Google Colab. Gom các ảnh từ ba thư mục rồi chia lại theo nhãn với tỷ lệ khoảng 70/20/10. |
| [resnet152v2.ipynb](./resnet152v2.ipynb) | ResNet152V2 với đầu phân loại sigmoid một đầu ra | Kaggle. Khối huấn luyện ResNet dùng trực tiếp các thư mục `train`, `val` và `test` của bộ dữ liệu. |

Cả hai mô hình khởi tạo backbone bằng trọng số ImageNet và dùng ảnh đầu vào RGB kích thước **224 × 224**.

## Chuẩn bị dữ liệu và môi trường

Tải bộ [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) của Paul Mooney. Chọn thư mục `chest_xray` trực tiếp chứa các tập con theo cấu trúc dưới đây. Sau khi giải nén, có thể có hai cấp thư mục cùng tên `chest_xray`.

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Mở notebook trên Colab hoặc Kaggle và bật GPU nếu huấn luyện. Cài các gói còn thiếu trong môi trường notebook.

```python
%pip install tensorflow opencv-python lime scikit-image numpy pandas scikit-learn matplotlib seaborn Pillow requests pydot
```

Khối EfficientNetB0 dùng `plot_model` để xuất sơ đồ kiến trúc. Nếu môi trường thiếu Graphviz, có thể bỏ qua lệnh vẽ sơ đồ trong `build_custom_model`.

## Huấn luyện và suy luận

### EfficientNetB0 trên Colab

1. Gắn Google Drive bằng `drive.mount("/content/drive")` sau khi import `drive` từ `google.colab`. Đặt `base_dir` trỏ đến thư mục dữ liệu.
2. Chạy phần tạo DataFrame, chia dữ liệu, tạo generator RGB, tính trọng số lớp và định nghĩa `build_custom_model`. Bắt đầu với khối `model_EfficientNetB0 = build_custom_model(...)` để huấn luyện mô hình với backbone đóng băng.
3. Nếu chạy tiếp khối huấn luyện toàn bộ mô hình, đổi `color_mode='grayscale'` ở ba generator của khối đó thành `color_mode='rgb'` để khớp đầu vào `(224, 224, 3)`.
4. Đổi các đường dẫn `ModelCheckpoint` dưới `/kaggle/working/` thành đường dẫn có quyền ghi trên Colab. Sau khi huấn luyện, trỏ các lệnh `load_model` đến đúng tệp `EfficientNetB0.keras`, `EfficientNetB0_full_model.keras` hoặc bản `.h5` tương ứng mà bạn đã lưu. Cập nhật `img_path` thành đường dẫn ảnh muốn thử.

### ResNet152V2 trên Kaggle

1. Thêm bộ dữ liệu vào notebook và cập nhật `dataset_dir`. Nếu chạy phần khảo sát dữ liệu ở đầu notebook, cập nhật cả `base_dir`.
2. Chạy phần tạo `train_generator`, `val_generator` và `test_generator`, rồi huấn luyện đầu phân loại. Có thể tiếp tục với cell fine-tuning. Khối này mở khóa các lớp có tên chứa `conv5_block1` và giảm learning rate.
3. Kiểm thử bằng `test_generator`. Để chạy demo, đổi đường dẫn nạp sang checkpoint `resnet152v2_pneumonia_best.h5` vừa tạo hoặc tệp mô hình tương ứng của bạn. Khi chỉ suy luận, có thể dùng `load_model(model_path, compile=False)` để nạp mô hình có metric tùy chỉnh.

Các cell demo cần tệp mô hình đã lưu sau huấn luyện hoặc checkpoint bạn đã có. Những đường dẫn Google Drive và Kaggle trong notebook là đường dẫn của các lần thực nghiệm.

## Giải thích dự đoán

| Phương pháp | Kết quả trực quan |
| --- | --- |
| Grad-CAM | Bản đồ nhiệt từ gradient tại lớp tích chập, chồng lên ảnh gốc. Notebook chọn `top_conv` cho EfficientNetB0 và `conv5_block3_3_conv` cho ResNet152V2. |
| LIME | Các nhóm điểm ảnh liền kề (*superpixel*) có đóng góp vào dự đoán của một ảnh cụ thể. |

Để giải thích cùng một dự đoán với hai phương pháp, giữ nhất quán checkpoint, thứ tự nhãn và tiền xử lý.

- **EfficientNetB0.** Generator RGB truyền giá trị pixel theo thang `[0, 255]` vào backbone có bước rescale tích hợp. Nếu giữ ảnh LIME ở `[0, 1]` để chia superpixel, nhân lại với `255` ngay trước cả lệnh dự đoán ban đầu và lệnh dự đoán trong `classifier_fn`. Cell Grad-CAM hiện tính gradient từ `effnet.output`, là vector đặc trưng. Để tạo heatmap theo lớp `NORMAL` hoặc `PNEUMONIA`, truyền vector đó qua các lớp phân loại sau backbone ngay bên trong `GradientTape` rồi lấy gradient theo điểm của lớp cần giải thích. [Tiền xử lý EfficientNet trong Keras](https://keras.io/api/applications/efficientnet/efficientnet_preprocessing/)
- **ResNet152V2.** Checkpoint được huấn luyện bằng generator chia pixel cho `255`. Khi dùng checkpoint này, áp dụng cùng phép chia trong `load_and_prep` của cell XAI và truyền trực tiếp các ảnh đã chuẩn hóa vào `model.predict` trong hàm LIME. Hàm này chuyển đầu ra sigmoid `p` thành hai xác suất `[1 - p, p]` theo thứ tự `NORMAL`, `PNEUMONIA`.

## Kết quả đã lưu

| Mô hình | Tập đánh giá | Accuracy |
| --- | --- | ---: |
| EfficientNetB0, backbone đóng băng | Validation chia lại, 1.171 ảnh đã xác thực | 92,40% |
| EfficientNetB0, khối huấn luyện toàn bộ mô hình | Test chia lại | 98,17% |
| ResNet152V2 sau fine-tuning | Thư mục test của bộ dữ liệu, 624 ảnh | 89,74% |

Các giá trị được ghi lại từ output có sẵn trong notebook. Hai mô hình sử dụng cách chia dữ liệu khác nhau nên bảng mô tả từng lần thực nghiệm. [Báo cáo đồ án](./doan.pdf) trình bày ví dụ dự đoán, bản đồ nhiệt Grad-CAM và ảnh LIME ở Chương IV.

## Tài liệu và tác giả

- [Báo cáo đồ án (PDF)](./doan.pdf)
- [Slide thuyết trình (PowerPoint)](./doancs_final.pptx)

**Sinh viên thực hiện** Nguyễn Đức Trọng

**Giảng viên hướng dẫn** ThS. Trần Hồ Đạt
