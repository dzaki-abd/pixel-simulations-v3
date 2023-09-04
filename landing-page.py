import streamlit as st
from mulai import show as show_mulai

def home(session_state):
    st.set_page_config(page_title="PixelSimulations", page_icon=":computer:", layout="wide")
    st.markdown("""
        <style>
        .title-center {
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Menampilkan judul di tengah halaman menggunakan st.markdown
    st.markdown("<h1 class='title-center'>PixelSimulations</h1>", unsafe_allow_html=True)

    # Mengatur tata letak untuk menempatkan tombol di tengah halaman secara horizontal
    col1, col2 = st.columns(2)
    with col1:  # Menggunakan with untuk mengatur tata letak tombol di tengah kolom kiri
        st.write("")  # Kolom kiri kosong untuk menengahkan tombol

    # Menggunakan CSS untuk mengatur tombol berada di tengah dan sejajar
    st.markdown("""
        <style>
        .stButton > button {
            display: block;
            width: 100%;
            margin: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.button("Start", key="mulai_btn"):
        # Jika tombol "Mulai" ditekan, arahkan ke halaman Mulai dari file mulai.py
        session_state["page"] = "mulai"

    with col2:  # Menggunakan with untuk mengatur tata letak tombol di tengah kolom kanan
        st.write("")  # Kolom kanan kosong untuk menengahkan tombol

    # if st.button("Info", key="info_btn"):
    #     # Jika tombol "Info" ditekan, arahkan ke halaman informasi
    #     session_state["page"] = "info"

def dashboard():
    st.header("Dashboard")
    st.write("Ini adalah halaman dashboard dari aplikasi.")
    # Tambahkan komponen dashboard lainnya di sini...

def info():
    st.header("Informasi")
    st.write("Ini adalah halaman informasi dari aplikasi.")
    st.write("Pada aplikasi ini, Anda dapat melakukan simulasi konvolusi pada gambar.")
    st.write("Fitur-fitur lainnya akan ditambahkan di sini...")
    # Tambahkan informasi lebih lanjut sesuai kebutuhan

def main():
    # Inisialisasi objek SessionState dengan nilai default untuk halaman saat ini
    session_state = st.session_state

    # Menampilkan halaman yang sesuai berdasarkan status session_state
    if session_state.get("page") == "mulai":
        show_mulai(session_state)
    else:
        home(session_state)

if __name__ == "__main__":
    main()
