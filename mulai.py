import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from skimage.feature import hog
from skimage import exposure
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte
import dlib
import tempfile


# Grayscale
def grayscale_image(image):
    grayscale_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return Image.fromarray(cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB))


# Histogram
def compute_histogram(image):
    grayscale_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([grayscale_img], [0], None, [256], [0, 256])
    return grayscale_img, hist

# Histogram Equalization
def histogram_equalization(image):
    image_array = np.array(image)
    r, g, b = cv2.split(image_array)

    r_equalized = cv2.equalizeHist(r)
    g_equalized = cv2.equalizeHist(g)
    b_equalized = cv2.equalizeHist(b)

    equalized_image = cv2.merge([r_equalized, g_equalized, b_equalized])

    # Hitung histogram setelah equalization
    equalized_r_hist = cv2.calcHist([r_equalized], [0], None, [256], [0, 256])
    equalized_g_hist = cv2.calcHist([g_equalized], [0], None, [256], [0, 256])
    equalized_b_hist = cv2.calcHist([b_equalized], [0], None, [256], [0, 256])

    return Image.fromarray(equalized_image), equalized_r_hist, equalized_g_hist, equalized_b_hist



# Contrast Stretching
def contrast_stretching(image):
    img_array = np.array(image)
    img_min = img_array.min()
    img_max = img_array.max()
    stretched_img = 255 * ((img_array - img_min) / (img_max - img_min))
    return Image.fromarray(stretched_img.astype("uint8"))


# Brightness
def brightness_adjustment(image, value):
    enhancer = ImageEnhance.Brightness(image)
    output_img = enhancer.enhance(value)
    return output_img


# Image Interpolation
def resize_image(image):
    width, height = image.size
    new_width = st.number_input(
        "Masukkan lebar gambar baru:", min_value=1, max_value=width, step=1, value=width
    )
    new_height = st.number_input(
        "Masukkan tinggi gambar baru:",
        min_value=1,
        max_value=height,
        step=1,
        value=height,
    )

    resized_img_linear = image.resize((new_width, new_height), resample=Image.LINEAR)
    resized_img_bilinear = image.resize(
        (new_width, new_height), resample=Image.BILINEAR
    )
    resized_img_bicubic = image.resize((new_width, new_height), resample=Image.BICUBIC)

    return resized_img_linear, resized_img_bilinear, resized_img_bicubic


# Black and White
def binarize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return binary_image


# Negative Image
def negative_image(image):
    negative_img = 255 - image
    return negative_img


# Image Filtering
def image_filtering(image, filter_type, kernel_size):
    # Menerapkan filter sesuai dengan jenis yang dipilih
    if filter_type == "Mean Filter":
        filtered_image = cv2.blur(image, (kernel_size, kernel_size))
    elif filter_type == "Gaussian Filter":
        filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_type == "Median Filter":
        filtered_image = cv2.medianBlur(image, kernel_size)
    elif filter_type == "Max Filter":
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        filtered_image = cv2.dilate(image, kernel, iterations=1)
    elif filter_type == "Min Filter":
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        filtered_image = cv2.erode(image, kernel, iterations=1)
    else:
        st.error("Filter yang dipilih tidak valid!")
        return None

    return filtered_image


# Image Sharpening
def image_sharpening(image, filter_type):
    # Inisialisasi filter
    if filter_type == "Laplacian Filter":
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    elif filter_type == "Sobel Filter":
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif filter_type == "Prewitt Filter":
        kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    elif filter_type == "Roberts Filter":
        kernel = np.array([[1, 0], [0, -1]])
    else:
        print("Filter yang dipilih tidak valid!")
        return None

    # Mengaplikasikan filter pada citra
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image


# Edge Detection
def edge_detection_image(image, filter):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    if filter == "Canny":
        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, 100, 200) / 255.0  # Normalize to [0.0, 1.0]
    elif filter == "Sobel":
        # Apply Sobel edge detection
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        edges = np.sqrt(sobel_x**2 + sobel_y**2) / 255.0  # Normalize to [0.0, 1.0]
    elif filter == "Prewitt":
        # Apply Prewitt edge detection
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewitt_x = cv2.filter2D(gray_image, -1, kernelx)
        prewitt_y = cv2.filter2D(gray_image, -1, kernely)
        edges = (
            np.sqrt(prewitt_x**2 + prewitt_y**2) / 255.0
        )  # Normalize to [0.0, 1.0]
    elif filter == "Scharr":
        # Apply Scharr edge detection
        scharr_x = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)
        edges = (
            np.sqrt(scharr_x**2 + scharr_y**2) / 255.0
        )  # Normalize to [0.0, 1.0]
    elif filter == "Laplacian":
        # Apply Laplacian edge detection
        edges = cv2.Laplacian(gray_image, cv2.CV_64F)
        edges = (edges + 255) / 510.0  # Normalize to [0.0, 1.0]

    return edges


# Global Thresholding
def global_tresholding(image, value):
    grayscale_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(grayscale_img, value, 255, cv2.THRESH_BINARY)
    return Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))


# Otsu Thresholding
def otsu_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


# Color Moment
def calculate_color_moment(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv_image[:, :, 0])
    s_mean = np.mean(hsv_image[:, :, 1])
    v_mean = np.mean(hsv_image[:, :, 2])
    return h_mean, s_mean, v_mean


# Color Histogram
def compute_rgb_histogram(image):
    r, g, b = cv2.split(np.array(image))
    r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
    b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
    return r_hist, g_hist, b_hist


# Histogram Orientation Gradient
def calculate_hog(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features, hog_image = hog(gray_image, visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_image_rescaled


# Local Binary Pattern
def calculate_lbp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp_image = local_binary_pattern(gray_image, n_points, radius, method="uniform")
    lbp_image = (lbp_image * 255).astype(np.uint8)  # Mengubah nilai piksel ke kisaran 0-255
    return lbp_image

# Live Camera
def livecamera():
    # st.set_page_config(page_title="Streamlit WebCam App")
    # st.title("Webcam Display Streamlit App with Face Detection")
    # st.caption("Powered by OpenCV, Streamlit")
    
    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    # col1, col2 = st.columns(2)  # Create two columns
    # stop_button_pressed = col1.button("Berhenti")
    # start_button_pressed = col2.button("Mulai")
    stop_button_pressed = None

    while cap.isOpened() and not stop_button_pressed:
        ret_val, frame = cap.read()
        if not ret_val:
            st.write("Video Capture Ended")
            break
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Convert the frame back to RGB for displaying with Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
        
        if cv2.waitKey(1) == 13 or stop_button_pressed:
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Face Detection - for swapping faces
def face_detection():
    ## Our pretrained model that predicts the rectangles that correspond to the facial features of a face
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    SCALE_FACTOR = 1
    FEATHER_AMOUNT = 11
    FACE_POINTS = list(range(17, 68))
    MOUTH_POINTS = list(range(48, 61))
    RIGHT_BROW_POINTS = list(range(17, 22))
    LEFT_BROW_POINTS = list(range(22, 27))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    NOSE_POINTS = list(range(27, 35))
    JAW_POINTS = list(range(0, 17))

    # Points used to line up the images.
    ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                    RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

    # Points from the second image to overlay on the first. The convex hull of each
    # element will be overlaid.
    OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS +
                    RIGHT_BROW_POINTS, NOSE_POINTS + MOUTH_POINTS]

    # Amount of blur to use during colour correction, as a fraction of the
    # pupillary distance.
    COLOUR_CORRECT_BLUR_FRAC = 0.6

    #menggunakan haar cascade
    cascade_path = 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_path)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    frame_placeholder = st.empty()

    #mendapakan landmark
    def get_landmarks(im, dlibOn):
        if (dlibOn == True):
            rects = detector(im, 1)
            if len(rects) > 1:
                # return "error"
                return None
            if len(rects) == 0:
                # return "error"
                return None
            return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
        else:
            rects = cascade.detectMultiScale(im, 1.3, 5)
            if len(rects) > 1:
                # return "error"
                return None
            if len(rects) == 0:
                # return "error"
                return None

            x, y, w, h = rects[0]
            rect = dlib.rectangle(x, y, x + w, y + h)
            return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

    #untuk menggambar poligon cembung (convex hull) di sekitar sekelompok titik landmark
    def draw_convex_hull(im, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)

    def get_face_mask(im, landmarks):
        im = np.zeros(im.shape[:2], dtype=np.float64)

        for group in OVERLAY_POINTS:
            draw_convex_hull(im, landmarks[group], color=1)

        im = np.array([im, im, im]).transpose((1, 2, 0))
        im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
        im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

        return im

    #mulai menjalankan swapping dengan menandakan koordinat landmark
    def transformation_from_points(points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(points1.T * points2)

        R = (U * Vt).T

        return np.vstack([np.hstack(((s2 / s1) * R,
                                        c2.T - (s2 / s1) * R * c1.T)),
                            np.matrix([0., 0., 1.])])

    def read_im_and_landmarks(fname):
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        im = cv2.resize(im, None, fx=0.35, fy=0.35,
                        interpolation=cv2.INTER_LINEAR)
        im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                            im.shape[0] * SCALE_FACTOR))
        s = get_landmarks(im, dlibOn)

        return im, s

    def warp_im(im, M, dshape):
        output_im = np.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im,
                    M[:2],
                    (dshape[1], dshape[0]),
                    dst=output_im,
                    borderMode=cv2.BORDER_TRANSPARENT,
                    flags=cv2.WARP_INVERSE_MAP)

        return output_im

    def correct_colours(im1, im2, landmarks1):
        blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
            np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
            np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1

        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

        # Avoid divide-by-zero errors.
        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

        return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                im2_blur.astype(np.float64))

    def face_swap(img, name):
        s = get_landmarks(img, True)
        if s is None:
            # print("No or too many faces")
            return img
        im1, landmarks1 = img, s
        im2, landmarks2 = read_im_and_landmarks(name)

        M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                    landmarks2[ALIGN_POINTS])
        mask = get_face_mask(im2, landmarks2)
        warped_mask = warp_im(mask, M, im1.shape)
        combined_mask = np.max([get_face_mask(im1, landmarks1),
                                warped_mask],
                                axis=0)
        warped_im2 = warp_im(im2, M, im1.shape)
        warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

        output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

        # output_im is no longer in the expected OpenCV format so we use openCV
        # to write the image to diks and then reload it
        temp_output_path = tempfile.mktemp(suffix=".jpg")
        cv2.imwrite(temp_output_path, output_im)
        image = cv2.imread(temp_output_path)
        # cv2.imwrite('output.jpg', output_im)
        # image = cv2.imread('output.jpg')
        # frame = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

        return image

    cap = cv2.VideoCapture(0)
    # filter_image = 'obama.jpg'
    dlibOn = False

    uploaded_image = st.file_uploader("Upload image with face (PNG or JPEG)", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_image:
            temp_image.write(uploaded_image.read())
            filter_image = temp_image.name

        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            frame = cv2.flip(frame, 1)
            swapped_frame = face_swap(frame, filter_image)
            swapped_frame = cv2.cvtColor(swapped_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(swapped_frame, channels="RGB")
            if cv2.waitKey(1) == 13:  # 13 is the Enter Key
                break
    else:
        st.write("Information : Upload image with face to swap")
        st.empty()

    cap.release()
    cv2.destroyAllWindows()


# Other Effect
def other_effect(image, effect):
    if effect == "Emboss":
        # Create the emboss kernel
        kernel = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]], dtype=np.float32)

        # Ensure the image is in numpy array format
        image = np.array(image)

        # Apply the emboss kernel to the image
        emboss_image = cv2.filter2D(image, -1, kernel)

        # Normalize the image to [0, 255]
        emboss_image = np.clip(emboss_image + 128, 0, 255).astype(np.uint8)

        return emboss_image

    elif effect == "Sepia":
        # Create the sepia filter matrix
        sepia_filter = np.array(
            [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
        )

        # Ensure the image is in numpy array format
        image = np.array(image)

        # Apply the sepia filter to the image
        sepia_image = cv2.transform(image, sepia_filter)

        # Normalize the image to [0, 255]
        sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)

        return sepia_image


def show(session_state):
    st.set_page_config(page_title="PixelSimulations", page_icon=":computer:", layout="wide")
    st.title("PixelSimulations")

    # Mengatur tata letak dengan 2 kolom di sidebar
    col1, col2 = st.sidebar.columns([1, 1])
    
    logo_kiri = "img/itenas-logo.png"
    with col1:
        st.image(logo_kiri, use_column_width=False, caption="", width=100)
        st.markdown(
            "<style>.stImage > img {display: block;margin: 0 auto;}</style>",
            unsafe_allow_html=True,
        )

    # Tambahkan logo kanan di sidebar (kolom 2)
    logo_kanan = "img/kampus-merdeka.png"  # Ganti dengan path atau URL gambar logo kanan Anda
    with col2:
        st.image(logo_kanan, use_column_width=False, caption="", width=100)
        st.markdown(
            "<style>.stImage > img {display: block;margin: 0 auto;}</style>",
            unsafe_allow_html=True,
        )

    menu = {
        "Pre-processing": [
            "Grayscale",
            "Histogram",
            "Histogram Equalization",
            "Contrast Stretching",
            "Brightness Adjustment",
            "Image Interpolation",
            "Black and White",
            "Negative Image",
            "Image Filtering",
            "Edge Detection",
            "Image Sharpening",
        ],
        "Image Segmentation & Feature Extraction": [
            "Global Thresholding",
            "Otsu Thresholding",
            "Color Moment",
            "Color Histogram",
            "Histogram Orientation Gradient",
            "Local Binary Pattern",
        ],
        # "Image Classification": [
        #     "K-Nearest Neighbor",
        #     "K-Means",
        # ],
        "Face Detection": [
            "Live Camera",
            "Swap Face",
            
        ],
        "Other": [
            "Emboss",
            "Sepia",
        ],
    }
    selected_category = st.sidebar.selectbox("Select Category", list(menu.keys()))
    choice = st.sidebar.selectbox("Select Process", menu[selected_category])

    back_button = st.sidebar.button("Kembali")
    if back_button:
        session_state["page"] = None

    if choice == "Grayscale":
        st.header("Grayscale")
        uploaded_image = st.file_uploader(
            "Upload citra berwarna...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_column_width=True)

            grayscale_img = grayscale_image(image)
            st.image(grayscale_img, caption="Hasil grayscale", use_column_width=True)

    elif choice == "Histogram":
        st.header("Histogram")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_column_width=True)

            grayscale_img, hist = compute_histogram(image)

            # Tampilkan hasil citra grayscale
            st.image(grayscale_img, caption="Hasil histogram grayscale", use_column_width=True)

            # Tampilkan histogram
            plt.figure(figsize=(8, 6))
            plt.title("Histogram Grayscale")
            plt.xlabel("Intensitas Piksel")
            plt.ylabel("Jumlah Piksel")
            plt.plot(hist)
            st.pyplot(plt)

    elif choice == "Histogram Equalization":
        st.header("Histogram Equalization")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_column_width=True)

            equalized_img, r_hist, g_hist, b_hist = histogram_equalization(image)
            st.image(
                equalized_img,
                caption="Hasil histogram equalization",
                use_column_width=True,
            )

            # Tampilkan histogram RGB setelah equalization
            plt.figure(figsize=(8, 6))
            plt.title("Histogram RGB Setelah Equalization")
            plt.xlabel("Intensitas Piksel")
            plt.ylabel("Jumlah Piksel")
            plt.plot(r_hist, color="red", label="R")
            plt.plot(g_hist, color="green", label="G")
            plt.plot(b_hist, color="blue", label="B")
            plt.legend()
            st.pyplot(plt)

    elif choice == "Contrast Stretching":
        st.header("Contrast Stretching")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_column_width=True)

            stretched_img = contrast_stretching(image)
            st.image(
                stretched_img,
                caption="Hasil contrast stretching",
                use_column_width=True,
            )

    elif choice == "Brightness Adjustment":
        st.header("Brightness Adjustment")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_column_width=True)

            # add slider to adjust brightness
            value = st.slider("Brightness", -255, 255, 0, 1)

            output_img = brightness_adjustment(image, value)

            st.image(
                output_img, caption=f"Hasil Brightness Adjustment Dengan Nilai {value}", use_column_width=True
            )

    elif choice == "Image Interpolation":
        st.header("Image Interpolation")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_column_width=True)

            (
                resized_img_linear,
                resized_img_bilinear,
                resized_img_bicubic,
            ) = resize_image(image)

            st.subheader("Interpolasi Linear")
            st.image(
                resized_img_linear,
                caption="Hasil Interpolasi Linear",
                use_column_width=True,
            )

            st.subheader("Interpolasi Bilinear")
            st.image(
                resized_img_bilinear,
                caption="Hasil Interpolasi Bilinear",
                use_column_width=True,
            )

            st.subheader("Interpolasi Bicubic")
            st.image(
                resized_img_bicubic,
                caption="Hasil Interpolasi Bicubic",
                use_column_width=True,
            )

    elif choice == "Black and White":
        st.header("Black and White")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            image_array = np.array(image)  # Mengubah gambar menjadi array numpy

            st.image(image, caption="Original Image", use_column_width=True)

            output_img = binarize_image(image_array)
            st.image(
                output_img, caption="Hasil Black and White Image", use_column_width=True
            )

    elif choice == "Negative Image":
        st.header("Negative Image")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            image_array = np.array(image)  # Mengubah gambar menjadi array numpy

            st.image(image, caption="Original Image", use_column_width=True)

            negative_img = negative_image(image_array)
            st.image(
                negative_img, caption="Hasil Negative Image", use_column_width=True
            )

    elif choice == "Image Filtering":
        st.header("Image Filtering")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            image = np.array(image)

            st.image(image, caption="Original Image", use_column_width=True)

            kernel_size = st.slider("Ukuran kernel", 3, 15, 3, 2)
            filter_type = st.selectbox(
                "Tipe filter",
                [
                    "Mean Filter",
                    "Gaussian Filter",
                    "Median Filter",
                    "Max Filter",
                    "Min Filter",
                ],
            )
            
            filtered_img = image_filtering(image, filter_type, kernel_size)

            st.image(
                filtered_img, caption=f"Hasil {filter_type}", use_column_width=True
            )

    elif choice == "Image Sharpening":
        st.header("Image Sharpening")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            image = np.array(image)

            st.image(image, caption="Original Image", use_column_width=True)

            filter_type = st.selectbox(
                "Tipe filter",
                [
                    "Laplacian Filter",
                    "Sobel Filter",
                    "Prewitt Filter",
                    "Roberts Filter",
                ],
            )
            
            filtered_img = image_sharpening(image, filter_type)

            st.image(
                filtered_img, caption=f"Hasil {filter_type}", use_column_width=True
            )

    elif choice == "Edge Detection":
        st.header("Edge Detection")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_column_width=True)

            filter_type = st.selectbox(
                "Tipe filter", ["Canny", "Sobel", "Prewitt", "Scharr", "Laplacian"]
            )

            edge_detection_img = edge_detection_image(image, filter_type)

            st.image(
                edge_detection_img,
                caption=f"Hasil {filter_type} Edge Detection",
                use_column_width=True,
                clamp=True,
            )

    elif choice == "Global Thresholding":
        st.header("Global Thresholding")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_column_width=True)

            # add slider to adjust trheshold
            value = st.slider("Threshold", -255, 255, 0, 1)

            output_img = global_tresholding(image, value)

            st.image(
                output_img, caption=f"Hasil Global Thresholding Dengan Nilai Threshold {value}", use_column_width=True
            )

    elif choice == "Otsu Thresholding":
        st.header("Otsu Thresholding")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )        

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            image_array = np.array(image)  # Mengubah gambar menjadi array numpy

            st.image(image, caption="Original Image", use_column_width=True)

            otsu_img = otsu_threshold(image_array)
            st.image(
                otsu_img, caption="Hasil Otsu Thresholding", use_column_width=True
            )
    
    elif choice == "Color Moment":
        st.header("Color Moment")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            image_array = np.array(image)  # Mengubah gambar menjadi array numpy

            st.image(image, caption="Original Image", use_column_width=True)

            h_mean, s_mean, v_mean = calculate_color_moment(image_array)
            st.write(f"Hue Mean: {h_mean:.2f}")
            st.write(f"Saturation Mean: {s_mean:.2f}")
            st.write(f"Value Mean: {v_mean:.2f}")

    elif choice == "Color Histogram":
        st.header("Color Histogram")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_column_width=True)

            # Konversi gambar PNG ke format JPEG
            if image.format == "PNG":
                jpeg_image = Image.new("RGB", image.size)
                jpeg_image.paste(image)
                image = jpeg_image

            image_array = np.array(image)
            num_channels = image_array.shape[2] if len(image_array.shape) == 3 else 1

            r_hist, g_hist, b_hist = compute_rgb_histogram(image_array)

            plt.figure(figsize=(8, 6))
            plt.title("Histogram RGB")
            plt.xlabel("Intensitas Piksel")
            plt.ylabel("Jumlah Piksel")
            plt.plot(r_hist, color="red", label="R")
            plt.plot(g_hist, color="green", label="G")
            plt.plot(b_hist, color="blue", label="B")
            plt.legend()
            st.pyplot(plt)

    elif choice == "Histogram Orientation Gradient":
        st.header("Histogram Orientation Gradient")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            image_array = np.array(image)  # Mengubah gambar menjadi array numpy

            st.image(image, caption="Original Image", use_column_width=True)

            hog_img = calculate_hog(image_array)
            st.image(
                hog_img, caption="Hasil Histogram of Oriented Gradients", use_column_width=True
            )

    elif choice == "Local Binary Pattern":
        st.header("Local Binary Pattern")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            image_array = np.array(image)  # Mengubah gambar menjadi array numpy

            st.image(image, caption="Original Image", use_column_width=True)

            lbp_img = calculate_lbp(image_array)
            st.image(
                lbp_img, caption="Hasil Local Binary Pattern", use_column_width=True, channels="GRAY"
            )

    elif choice == "Emboss":
        st.header("Emboss Filter")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_column_width=True)

            # filter_type = st.selectbox("Tipe filter", ["Emboss", "Sepia"])
            filter_type = "Emboss"

            other_effect_img = other_effect(image, filter_type)

            st.image(
                other_effect_img,
                caption=f"Hasil {filter_type} Filter",
                use_column_width=True,
            )

    elif choice == "Sepia":
        st.header("Sepia Filter")
        uploaded_image = st.file_uploader(
            "Upload citra...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_column_width=True)

            filter_type = "Sepia"

            other_effect_img = other_effect(image, filter_type)

            st.image(
                other_effect_img,
                caption=f"Hasil {filter_type} Filter",
                use_column_width=True,
            )
    
    elif choice == "Live Camera":
        st.header("FD - Live Camera")
        livecamera()

    elif choice == "Swap Face":
        st.header("FD - Swap Face")
        face_detection()


if __name__ == "__main__":
    show()
