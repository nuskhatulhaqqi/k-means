import numpy as np 
from sklearn.cluster import KMeans
import streamlit as st 
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data_penduduk.csv', index_col=0)
ukuran = data.shape
XX=data.iloc[:,2:14].values
#preposesing
scaler = StandardScaler()
X = scaler.fit_transform(XX)

with st.sidebar:
    selected = option_menu(
        menu_title ="Main Menu",
        options = ['Home','Dataset','Implementasi','Uji SSE'], 
        menu_icon='case',
        default_index=0,
    )
if selected == 'Home':
    st.title("K- Means")
    st.write('''
    # Pengertian Klastering\n
    * Klastering adalah suatu pengelompokan data  seperti klasifikasi namun tidak memiliki label.\n
    # Pengertian K-Means\n
    * Metode ini digunakan untuk membangun kelompok dari objek-objek atau klister dimana objek dalam satu klister tertentu memiliki kesamaan ciri yang tinggi.\n
    * K-Mean clustering adalah metode untuk mengelompokkan item ke dalam suatu klister (dimana k adalah jumlah yang digunakan dalam klister).\n
    * Konsep dasar dari K-Means adalah pencarian pusat cluster secara iteratif.\n
    * Pusat cluster ditetapkan berdasarkan jarak setiap data ke pusat cluster.\n
    # Algoritma K-Means\n
    1. Tentukan jumlah cluster (K), tetapkan pusat cluster sembarang.\n
    2. Hitung jarak setiap data ke pusat cluster.\n
    3. Kelompokkan data ke dalam cluster yang dengan jarak yang paling pendek.\n
    4. Hitung pusat cluster.\n
    5. Ulangi langkah 2 - 4 hingga sudah tidak ada lagi data yang berpindah ke cluster yang lain.\n
    # Metode Menghitung Jarak\n
    * Penentuan jarak terdekat anatara data dengan centroid menggunakan rumus Euclidean Distance.\n
    ''')
    st.image('jarak.png')
    st.write('''
    n  	: jumlah dimensi/ atribut\n
    ak, bk 	: atribut ke k dari objek data, a = data dalam table , b = data centroid\n

    ''')

if selected == 'Dataset':
    st.title("Tentang Dataset")
    st.write('Dataset Jumlah Penduduk:')
    st.write(data)
    st.write('''
    Keterangan :\n
    Dataset yang kami gunakan dataset jumlah penduduk per-desa pada Kabupaten Bangkalan.
    Dataset ini terdiri dari 281 data dan terdapat 14 fitur.
    ''')
    st.write('Ukuran Data:')
    st.write(ukuran)
    st.write('Data yang di proses:')
    st.write(XX)
    st.write('Data setelah Preprosesing:')
    st.write(X)

if selected == 'Implementasi':
    st.title(f"{selected}")

# st.title(""" Web Apps Klastering K-Means \n""")

    def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
        """ 
        Create a plot of the covariance confidence elil
        
        """
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x,y)
        pearson = cov[0,1]/np.sqrt(cov[0,0]* cov[1,1])
        ell_radius_x = np.sqrt(1+pearson)
        ell_radius_y = np.sqrt(1-pearson)
        ellipse = Ellipse((0,0), width=ell_radius_x*2, height=ell_radius_y*2,facecolor=facecolor, **kwargs)
        scale_x = np.sqrt(cov[0,0]) * n_std
        mean_x = np.mean(x)
        scale_y = np.sqrt(cov[1,1]) * n_std
        mean_y = np.mean(y)
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
    
    def nilai(a):
        if a == 'Balita':
            h=0
        elif a == 'Anak-Anak':
            h=1
        elif a == 'Remaja Awal':
            h=2
        elif a == 'Remaja Akhir':
            h=3
        elif a == 'Dewasa Awal':
            h=4
        elif a == 'Dewasa Akhir':
            h=5
        elif a == 'Lansia Awal':
            h=6
        elif a == 'Lansia Akhir':
            h=7
        elif a == 'Manula Atas':
            h=8
        elif a == 'Jumlah KK':
            h=9
        elif a == 'laki-laki':
            h=10
        else:
            h=11
        return h

    klaster_slider = st.slider(min_value=1,max_value=10, value=2, label= " Jumlah Klaster")
    kmeans = KMeans(n_clusters=klaster_slider, random_state=2022).fit(X)
    labels = kmeans.labels_
    pilih_x = st.selectbox("Pilih Nilai x? ", ['Balita','Anak-Anak','Remaja Awal','Remaja Akhir','Dewasa Awal','Dewasa Akhir','Lansia Awal','Lansia Akhir','Manula Atas','Jumlah KK','laki-laki','perempuan'])
    
    pilih_y = st.selectbox("Pilih Nilai y? ", ['Balita','Anak-Anak','Remaja Awal','Remaja Akhir','Dewasa Awal','Dewasa Akhir','Lansia Awal','Lansia Akhir','Manula Atas','Jumlah KK','laki-laki','perempuan'])
    seleksi1 = st.selectbox("Visualisasi Batas Condence? ", [False,True])
    seleksi2 = st.selectbox("jumlah standar defiasi : ", [1,2,3])
    warna = ['red','seagreen','orange','blue','yellow','purple','green','brown','pink','lightblue']
    jumlah_label = len(set(labels))
    individu = st.selectbox("sunplot individu? ",[False, True])

    if individu:
        fig, ax = plt.subplots(ncols=jumlah_label)
    else:
        fig, ax = plt.subplots()

    for i, yi in enumerate(set(labels)):
        if not individu:
            a=ax
        else:
            a=ax[i]

        xi = X[labels == yi]
        x=nilai(pilih_x)
        y=nilai(pilih_y)
        x_pts = xi[:,x]
        y_pts = xi[:,y]
        a.scatter(x_pts,y_pts,c=warna[yi])
        plt.xlabel(pilih_x)
        plt.ylabel(pilih_y)
        if seleksi1:
            confidence_ellipse(x=x_pts, y=y_pts, ax=a ,edgecolor='black',facecolor=warna[yi],alpha=0.2, n_std=seleksi2)

    plt.tight_layout()
    st.write(fig)

if selected == 'Uji SSE':
    st.title("Uji Sum Of Squared Errors")
    klaster_slider = st.slider(min_value=1,max_value=30, value=2, label= "Batas Akhir Pengecekan: ")
    sse = []
    index = range(1,klaster_slider)
    for i in index:
        kmeans = KMeans(n_clusters=i, random_state=30)
        kmeans.fit(X)
        sse_ = kmeans.inertia_
        sse.append(sse_)

    fig2 = plt.figure()
    ax2 = fig2.add_axes([0,0,1,1])
    ax2.plot(index,sse)
    plt.xlabel('n_cluster')
    plt.ylabel('sse')
    plt.show()
    st.pyplot(fig2)
    st.write('''
    Keterangan :\n
    Untuk mengetahui nilai k terbaik cek nilai dimana letak awal grafik nya menurun secara horizontal atau penurunannya selisih sedikit.\n 
    Setelah dilakukan pengecekan hasilnya klaster yang paling bagus untuk di gunakan adalah klaster 6 atau k=6
    ''')



        
