import cv2
from ultralytics import YOLO
from pytube import YouTube

#def download_youtube_video(url, save_path="."):
#    yt = YouTube(url)
#    ys = yt.streams.get_highest_resolution()
#    ys.download(save_path)
#    return ys.default_filename

# Carregar o modelo YOLOv8 treinado
model = YOLO("yolov8n(1).pt")


# Insira o link do vídeo do YouTube aqui
#url = 'https://www.youtube.com/watch?v=-TH6V9360VQ'
#video_path = download_youtube_video(url)

video_path = "DJI_0502.MOV"

# Abrir o vídeo
cap = cv2.VideoCapture(video_path)


# Verificar se o vídeo foi aberto com sucesso
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Processar o vídeo quadro a quadro
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Fim do vídeo

    # Realizar a detecção de objetos no quadro
    results = model.predict(frame)

    # Filtrar as detecções para apenas carros (classe 'car' em COCO é 2)
    cars = [detection for detection in results[0].boxes if int(detection.cls) == 2]

    # Desenhar as caixas delimitadoras para os carros detectados
    for car in cars:
        box = car.xyxy.cpu().numpy()  # Transferir para a CPU e converter para NumPy
        if box.shape == (1, 4):
            x1, y1, x2, y2 = map(int, box[0])  # Acessar a primeira linha e converter para inteiros
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Exibir o quadro
    cv2.imshow("Detected Cars", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Carregar a imagem
#image_path = "carros3.jpg"
#image = cv2.imread(image_path)

# Realizar a detecção de objetos
#results = model.predict(image)

# Filtrar as detecções para apenas carros (classe 'car' em COCO é 2)
#cars = [detection for detection in results[0].boxes if int(detection.cls) == 2]

# Desenhar as caixas delimitadoras para os carros detectados
#for car in cars:
#    box = car.xyxy.cpu().numpy()  # Transferir para a CPU e converter para NumPy
#    if box.shape == (1, 4):
#        x1, y1, x2, y2 = map(int, box[0])  # Acessar a primeira linha e converter para inteiros
#        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#    else:
#        print("Formato inesperado para a caixa delimitadora:", box)

# Liberar recursos
cap.release()


# Exibir a imagem
#cv2.imshow("Detected Cars", image)
#cv2.waitKey(0)
cv2.destroyAllWindows()
