import asyncio
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from core import Core

# Global değişken: Haritayı kontrol etmek için bir bayrak
close_flag = False

def on_close(event):
    """Pencere kapatma olayı."""
    global close_flag
    close_flag = True  # Bayrağı ayarla

async def main(core: Core):
    """
    Bu fonksiyonun ismi ve argümanları değiştirilemez.
    Eğitilmiş yapay zeka modelinin girdi ve çıktılarını bu fonksiyonda burada işleyin.
    """

    global close_flag

    # Scaler ve KMeans modellerini yükleyin
    scaler = joblib.load("scaler.joblib")
    kmeans = joblib.load("kmeans_model.joblib")

    # Harita için matplotlib figürü oluşturun
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('close_event', on_close)  # Pencere kapatma olayını dinle

    # Başlangıçta robot pozisyonunu belirleyin
    robot_position = [0, 0]  # Robot başlangıç pozisyonu (0, 0)

    # Harita üzerine mesafe bölgelerini ekleyin (içi boş dairesel çizgiler)
    ax.add_patch(Circle(robot_position, 10, color='red', fill=False, linewidth=1.5, label="Yakın Mesafe (0-10 cm)"))
    ax.add_patch(Circle(robot_position, 50, color='orange', fill=False, linewidth=1.5, label="Orta Mesafe (10-50 cm)"))
    ax.add_patch(Circle(robot_position, 100, color='green', fill=False, linewidth=1.5, label="Uzak Mesafe (50+ cm)"))

    # Haritayı ayarlayın
    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
    ax.set_aspect('equal')

    # Sağ üstte sabit açıklamalar için legend ekle
    ax.legend(loc="upper right")

    # Önceki mesafe etiketi (aynı durumun tekrar işlenmesini önlemek için)
    previous_label = None

    # Girdi alma, tahmin ettirme ve motor çıkışı verme işlemlerini bu döngü içerisinde yapın
    while not close_flag:  # Pencere kapatılana kadar döngü çalışır
        # Ultrasonik mesafe verisini alın
        distance = await core.get_ultrasonic_distance()

        # Mesafe verisini modeli eğitirken kullandığınız formata çevirin ve ölçeklendirin
        data = np.array([[distance, 0]])  # Örnek olarak, mesafe ve bir sabit özellik (0) ekledik
        data_scaled = scaler.transform(data)

        # KMeans modeli ile tahmin yapın
        cluster_label = kmeans.predict(data_scaled)[0]  # Tahmin edilen küme etiketi

        # Tahmin edilen değere göre haritaya engel ekle
        obstacle_position = [distance, 0]
        if cluster_label == 0:
            color = 'red'
            label = "yakın"
        elif cluster_label == 1:
            color = 'orange'
            label = "orta"
        else:
            color = 'green'
            label = "uzak"

        # Haritayı güncelle
        ax.clear()  # Haritayı temizleyin

        # Mesafe bölgelerini tekrar çizin
        ax.add_patch(Circle(robot_position, 10, color='red', fill=False, linewidth=1.5))
        ax.add_patch(Circle(robot_position, 50, color='orange', fill=False, linewidth=1.5))
        ax.add_patch(Circle(robot_position, 100, color='green', fill=False, linewidth=1.5))

        # Robot pozisyonunu merkeze sabitle ve kırmızı olarak göster
        ax.plot(robot_position[0], robot_position[1], "ro", label="Robot")  # 'ro' kırmızı nokta

        # Engel pozisyonunu ekle
        ax.plot(obstacle_position[0], obstacle_position[1], "o", color=color, label=f"Engel {label}")

        # Harita sınırlarını ve oranlarını tekrar ayarlayın
        ax.set_xlim(-110, 110)
        ax.set_ylim(-110, 110)
        ax.set_aspect('equal')

        # Sabit açıklamaları tekrar çizmek için legend ekle
        ax.legend(loc="upper right", labels=[
            "Yakın Mesafe (0-10 cm)",
            "Orta Mesafe (10-50 cm)",
            "Uzak Mesafe (50+ cm)",
            "Robot"
        ])

        plt.draw()
        plt.pause(3.0)  # Haritayı güncelleme hızını ayarlayın

        # Robotun ekranına durumu yazdır
        await core.set_state("Mesafe durumu: " + label)

    plt.close(fig)  # Döngü bittiğinde figürü kapat

if __name__ == "__main__":
    """
    Bu kısımda kendi testlerinizi yazabilirsiniz. Robota gönderdiğinizde bu kısım çalışmayacaktır.
    """
    core = Core()
    asyncio.run(main(core))
