from ultralytics import YOLO

import sys
import cv2
import numpy as np
import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, 
                             QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
                             QGroupBox, QCheckBox, QScrollArea, QFileDialog, QComboBox,
                             QRadioButton, QButtonGroup)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

class PrototipoTesisVictoria(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # --- CARGAR MODELO YOLO ---
        self.modelo_yolo = YOLO('yolov8n.pt')
        # Diccionario de clases COCO que nos interesan
        self.clases_disponibles = {
            "Manzana": 47,
            "Plátano": 46,
            "Perro": 16,       
            "Gato": 17,        
            "Naranja": 49,
            "Persona": 0,
            "Teléfono": 67,
            "Laptop": 63,      
            "Ratón (Mouse)": 64, 
            "Teclado": 66,     
            "Taza": 41,        
            "Silla": 56,       
            "Botella": 39,
            "Mochila": 24,     
            "Tenedor": 42      
        }

        # --- VARIABLES DE ESTADO ---
        self.modo_actual = "COLOR" # Puede ser "COLOR" o "YOLO"
        self.elementos_activos = [] # Guarda tanto colores como clases YOLO
        self.ultimo_frame_raw = None 
        self.modo_imagen = False
        self.frame_estatico = None 
        self.modo_video = False # Variable para controlar si es un video grabado

        # --- CONFIGURACIÓN DE LA VENTANA ---
        # --- CONFIGURACIÓN DE LA VENTANA ---
        self.setWindowTitle("Detector por objetos y colores")
        self.setFixedSize(1050, 750) 
        self.setStyleSheet("QMainWindow { background-color: #2c3e50; }")

        # --- COMPONENTES DE LA INTERFAZ ---
        self.lbl_info = QLabel("Modo Activo: Detección por Color (HSV)")
        self.lbl_info.setStyleSheet("font-weight: bold; color: #ecf0f1; font-size: 16px; padding: 10px;")

        self.lbl_video = QLabel()
        self.lbl_video.setFixedSize(640, 480) 
        self.lbl_video.setStyleSheet("border: 4px solid #34495e; background-color: black; border-radius: 10px;")
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video.mousePressEvent = self.get_color_clic # Evento de clic siempre activo, pero condicionado

        # --- PANEL LATERAL ---
        # 1. Selector de Modo (Radio Buttons)
        self.grupo_modos = QGroupBox("Modo de Detección")
        self.grupo_modos.setStyleSheet("QGroupBox { color: #f1c40f; font-weight: bold; border: 1px solid #f1c40f; }")
        layout_modos = QVBoxLayout()
        
        self.radio_color = QRadioButton("DETECTOR COLORES")
        self.radio_color.setChecked(True) # Color por defecto
        self.radio_color.setStyleSheet("color: white; font-weight: bold;")
        
        self.radio_yolo = QRadioButton("DETECTOR OBJETOS")
        self.radio_yolo.setStyleSheet("color: white; font-weight: bold;")

        self.btn_group = QButtonGroup()
        self.btn_group.addButton(self.radio_color)
        self.btn_group.addButton(self.radio_yolo)
        self.radio_color.toggled.connect(self.cambiar_modo)

        layout_modos.addWidget(self.radio_color)
        layout_modos.addWidget(self.radio_yolo)
        self.grupo_modos.setLayout(layout_modos)

        # 2. Controles específicos de YOLO
        self.widget_controles_yolo = QWidget()
        layout_yolo = QHBoxLayout(self.widget_controles_yolo)
        layout_yolo.setContentsMargins(0,0,0,0)
        
        self.combo_clases = QComboBox()
        self.combo_clases.addItems(self.clases_disponibles.keys())
        self.combo_clases.setStyleSheet("padding: 5px; font-weight: bold;")
        
        self.btn_agregar = QPushButton("➕ Agregar Clase")
        self.btn_agregar.setStyleSheet("background-color: #8e44ad; color: white; font-weight: bold; padding: 5px; border-radius: 5px;")
        self.btn_agregar.clicked.connect(self.agregar_clase_yolo)
        
        layout_yolo.addWidget(self.combo_clases)
        layout_yolo.addWidget(self.btn_agregar)
        self.widget_controles_yolo.setVisible(False) # Oculto al inicio

        # 3. Controles específicos de Color 
        self.lbl_instruccion_color = QLabel("Haz clic en el video para capturar color")
        self.lbl_instruccion_color.setStyleSheet("color: #3498db; font-style: italic; font-weight: bold;")

        # 4. Historial (Scroll Area)
        self.panel_historial = QGroupBox("Lista de Detecciones")
        self.panel_historial.setStyleSheet("QGroupBox { color: white; font-weight: bold; border: 1px solid #7f8c8d; }")
        self.layout_historial_lista = QVBoxLayout()
        self.layout_historial_lista.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.scroll_widget = QWidget()
        self.scroll_widget.setLayout(self.layout_historial_lista)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_widget) 
        self.scroll_area.setFixedWidth(320)
        self.scroll_area.setStyleSheet("background-color: #34495e; border: none;")

        # --- BOTONES GLOBALES ---
        self.btn_cargar = QPushButton("CARGAR IMAGEN")
        self.btn_cargar.setStyleSheet("background-color: #f39c12; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.btn_cargar.clicked.connect(self.cargar_imagen_archivo)

        # Botón de Video
        self.btn_cargar_video = QPushButton("CARGAR VIDEO")
        self.btn_cargar_video.setStyleSheet("background-color: #9b59b6; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.btn_cargar_video.clicked.connect(self.cargar_video_archivo)

        self.btn_camara = QPushButton("VOLVER A CÁMARA")
        self.btn_camara.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.btn_camara.clicked.connect(self.activar_camara)

        self.btn_foto = QPushButton("CAPTURAR EVIDENCIA")
        self.btn_foto.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; padding: 12px; border-radius: 5px;")
        self.btn_foto.clicked.connect(self.tomar_foto)
        
        self.btn_limpiar = QPushButton("Limpiar Detecciones")
        self.btn_limpiar.setStyleSheet("background-color: #7f8c8d; color: white; border-radius: 5px; padding: 5px;")
        self.btn_limpiar.clicked.connect(self.limpiar_historial_completo)

        
        self.lbl_autor = QLabel("Elaborado por Victoria Elizabeth Juárez Morales - Maestría en Ingeniería")
        self.lbl_autor.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_autor.setStyleSheet("color: #bdc3c7; font-style: italic; font-size: 12px; padding: 8px;")
        
        # --- ENSAMBLAJE DE LAYOUTS ---
        layout_lateral = QVBoxLayout()
        layout_lateral.addWidget(self.btn_cargar)
        layout_lateral.addWidget(self.btn_cargar_video) # Añadimos el botón de video
        layout_lateral.addWidget(self.btn_camara)
        layout_lateral.addWidget(self.grupo_modos) 
        layout_lateral.addWidget(self.widget_controles_yolo) 
        layout_lateral.addWidget(self.lbl_instruccion_color) 
        layout_lateral.addWidget(self.scroll_area)
        layout_lateral.addWidget(self.btn_limpiar)

        layout_central = QHBoxLayout()
        layout_central.addWidget(self.lbl_video)
        layout_central.addLayout(layout_lateral)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.lbl_info)
        main_layout.addLayout(layout_central)
        main_layout.addWidget(self.btn_foto)
        main_layout.addWidget(self.lbl_autor) 

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Cámara y Timer
        self.cap = cv2.VideoCapture(0) 
        self.timer = QTimer() 
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30) 

    # --- LÓGICA DE CAMBIO DE MODO ---
    def cambiar_modo(self):
        self.limpiar_historial_completo() 
        
        if self.radio_color.isChecked():
            self.modo_actual = "COLOR"
            self.widget_controles_yolo.setVisible(False)
            self.lbl_instruccion_color.setVisible(True)
            self.lbl_info.setText("Modo Activo: Detección por Color (HSV)")
        else:
            self.modo_actual = "YOLO"
            self.widget_controles_yolo.setVisible(True)
            self.lbl_instruccion_color.setVisible(False)
            self.lbl_info.setText("Modo Activo: Inteligencia Artificial (YOLOv8)")

    # --- LÓGICA PARA AGREGAR ELEMENTOS (COLOR O YOLO) ---
    def get_color_clic(self, event):
        # SOLO hacer caso al clic si estamos en modo color
        if self.modo_actual != "COLOR" or self.ultimo_frame_raw is None:
            return

        x, y = int(event.position().x()), int(event.position().y())
        if 0 <= x < 640 and 0 <= y < 480:
            hsv_img = cv2.cvtColor(self.ultimo_frame_raw, cv2.COLOR_BGR2HSV)
            h, s, v = int(hsv_img[y, x][0]), int(hsv_img[y, x][1]), int(hsv_img[y, x][2])
            
            bajo = np.array([max(0, h - 15), 60, 60])
            alto = np.array([min(179, h + 15), 255, 255])
            
            preview_hsv = np.uint8([[[h, s, v]]])
            preview_bgr = cv2.cvtColor(preview_hsv, cv2.COLOR_HSV2BGR)[0][0]
            color_bgr = (int(preview_bgr[0]), int(preview_bgr[1]), int(preview_bgr[2]))
            
            nombre = f"Color {len(self.elementos_activos)+1}"
            self.crear_widget_lista("COLOR", nombre, color_bgr, bajo=bajo, alto=alto)

    def agregar_clase_yolo(self):
        nombre_clase = self.combo_clases.currentText()
        id_coco = self.clases_disponibles[nombre_clase]

        for item in self.elementos_activos:
            if item.get('id_coco') == id_coco: return # Evitar duplicados

        color_bgr = (int(np.random.randint(50, 255)), int(np.random.randint(50, 255)), int(np.random.randint(50, 255)))
        self.crear_widget_lista("YOLO", nombre_clase, color_bgr, id_coco=id_coco)

    def crear_widget_lista(self, tipo, nombre, color_bgr, bajo=None, alto=None, id_coco=None):
        fila_widget = QWidget()
        fila_layout = QHBoxLayout(fila_widget)
        fila_layout.setContentsMargins(5, 5, 5, 5)

        r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
        color_texto = 'white' if tipo == "COLOR" and color_bgr[1] < 130 else 'black'

        chk = QCheckBox(f"{nombre} | Buscando...")
        chk.setChecked(True)
        chk.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {color_texto}; font-weight: bold; border-radius: 8px; padding: 10px;")
        
        btn_del = QPushButton("✕")
        btn_del.setFixedSize(28, 28)
        btn_del.setStyleSheet("background-color: #c0392b; color: white; border-radius: 14px;")
        
        fila_layout.addWidget(chk)
        fila_layout.addWidget(btn_del)
        
        item_data = {
            'tipo': tipo, 'widget': fila_widget, 'checkbox': chk, 
            'nombre': nombre, 'color_bgr': color_bgr,
            'bajo': bajo, 'alto': alto, 'id_coco': id_coco
        }
        self.elementos_activos.append(item_data)
        btn_del.clicked.connect(lambda: self.eliminar_elemento(item_data))
        self.layout_historial_lista.addWidget(fila_widget)

    def eliminar_elemento(self, item):
        self.layout_historial_lista.removeWidget(item['widget'])
        item['widget'].deleteLater()
        if item in self.elementos_activos: self.elementos_activos.remove(item)

    def limpiar_historial_completo(self):
        for item in self.elementos_activos[:]: self.eliminar_elemento(item)

    # --- LÓGICA DE FUENTE DE DATOS ---
    def cargar_imagen_archivo(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Archivos de imagen (*.jpg *.jpeg *.png)")
        if ruta:
            img = cv2.imread(ruta)
            if img is not None:
                self.frame_estatico = cv2.resize(img, (640, 480))
                self.modo_imagen = True
                self.modo_video = False

    def cargar_video_archivo(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar Video", "", "Archivos de video (*.mp4 *.avi *.mov *.mkv)")
        if ruta:
            self.cap.release()
            self.cap = cv2.VideoCapture(ruta)
            self.modo_imagen = False
            self.modo_video = True

    def activar_camara(self):
        self.modo_imagen = False
        self.modo_video = False
        self.frame_estatico = None
        self.cap.release()
        self.cap = cv2.VideoCapture(0)

    def tomar_foto(self):
        if self.ultimo_frame_raw is not None:
            fecha = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"Evidencia_Victoria_{fecha}.jpg", self.ultimo_frame_raw)

    # --- LÓGICA DE PROCESAMIENTO PRINCIPAL ---
    def update_frame(self):
        def draw_text_with_outline(img, text, position, font, scale, color, thickness, outline_color=(0,0,0), outline_thickness=3):
            x, y = position
            cv2.putText(img, text, (x, y), font, scale, outline_color, thickness + outline_thickness, cv2.LINE_AA)
            cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

        if self.modo_imagen and self.frame_estatico is not None:
            frame = self.frame_estatico.copy()
        else:
            ret, frame = self.cap.read()
            if not ret: 
                # Hacer bucle si es un video pregrabado
                if self.modo_video:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                if not ret: return
                
            frame = cv2.resize(frame, (640, 480))
        
        self.ultimo_frame_raw = frame.copy()

        # Dependiendo del modo elegido, llamamos a la lógica correspondiente
        if self.modo_actual == "COLOR":
            frame = self.procesar_modo_color(frame, draw_text_with_outline)
        else:
            frame = self.procesar_modo_yolo(frame, draw_text_with_outline)

        # Mostrar en la interfaz
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_img, w_img, ch = rgb_frame.shape
        img_qt = QImage(rgb_frame.data, w_img, h_img, ch * w_img, QImage.Format.Format_RGB888)
        self.lbl_video.setPixmap(QPixmap.fromImage(img_qt))

    # --- LÓGICA COLOR ---
    def procesar_modo_color(self, frame, draw_fn):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        overlay = frame.copy()
        y_offset = 40
        hay_detecciones = False

        for item in self.elementos_activos:
            if not item['checkbox'].isChecked(): continue
            
            mask = cv2.inRange(hsv, item['bajo'], item['alto'])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
            contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            obj_count = 0
            for c in contornos:
                if cv2.contourArea(c) > 1500:
                    obj_count += 1
                    cv2.drawContours(frame, [c], -1, item['color_bgr'], 3)
                    x, y, w, h = cv2.boundingRect(c)
                    draw_fn(frame, f"Obj {obj_count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, item['color_bgr'], 2)

            item['checkbox'].setText(f"{item['nombre']} | Objetos: {obj_count}")
            
            if obj_count > 0:
                cv2.rectangle(overlay, (410, y_offset - 25), (630, y_offset + 10), (45, 52, 54), -1)
                draw_fn(frame, f"{item['nombre']}: {obj_count} DETECCIONES", (420, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, item['color_bgr'], 2)
                y_offset += 45
                hay_detecciones = True

        if hay_detecciones:
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        return frame

    # --- LÓGICA YOLO ---
    def procesar_modo_yolo(self, frame, draw_fn):
        clases_a_detectar = [item['id_coco'] for item in self.elementos_activos if item['checkbox'].isChecked()]

        if clases_a_detectar:
            resultados = self.modelo_yolo(frame, classes=clases_a_detectar, verbose=False)
            conteos = {id_coco: 0 for id_coco in clases_a_detectar}

            for result in resultados:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                    cls_id = int(box.cls[0])               
                    conf = float(box.conf[0])              

                    if conf > 0.4: 
                        conteos[cls_id] += 1
                        for item in self.elementos_activos:
                            if item['id_coco'] == cls_id:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), item['color_bgr'], 3)
                                draw_fn(frame, f"{item['nombre']} {int(conf*100)}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, item['color_bgr'], 2)

            for item in self.elementos_activos:
                if item['checkbox'].isChecked():
                    cantidad = conteos.get(item['id_coco'], 0)
                    item['checkbox'].setText(f"{item['nombre']} | Detectados: {cantidad}")
        return frame

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = PrototipoTesisVictoria()
    ventana.show()
    sys.exit(app.exec())