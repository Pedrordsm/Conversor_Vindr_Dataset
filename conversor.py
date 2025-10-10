import os
import pydicom
import cv2
import pandas as pd
import shutil

class DICOMToYOLO:
    def __init__(self, annotations_train, annotations_test, 
                 image_labels_train, image_labels_test,
                 dicom_train_path, dicom_test_path, output_dir):
        
        self.annotations_train = annotations_train
        self.annotations_test = annotations_test
        self.image_labels_train = image_labels_train
        self.image_labels_test = image_labels_test
        self.dicom_train_path = dicom_train_path
        self.dicom_test_path = dicom_test_path
        self.output_dir = output_dir
        
        # Criar pastas do YOLO
        self.create_folders()
        
    def create_folders(self):

        folders = ['images/train', 'images/val', 'images/test',
                  'labels/train', 'labels/val', 'labels/test']
        
        for folder in folders:
            os.makedirs(os.path.join(self.output_dir, folder), exist_ok=True)
        print("Pastas criadas!")
    
    def load_csvs(self):

        # Carregar annotations
        ann_train = pd.read_csv(self.annotations_train)
        ann_test = pd.read_csv(self.annotations_test)
        
        # Carregar image labels
        img_train = pd.read_csv(self.image_labels_train)
        img_test = pd.read_csv(self.image_labels_test)
        
        #mapeamento de classes
        classes = ['Aortic enlargement','Atelectasis','Cardiomegaly','Calcification','Clavicle fracture','Consolidation','Edema','Emphysema',
                   'Enlarged PA','Interstitial lung disease(ILD)','Infiltration','Lung cavity','Lung cyst','Lung opacity','Mediastinal shift',
                   'Nodule/Mass','Pulmonary fribosis','Pneumothorax','Pleural thickening','Pleural effusion','Rib fracture','Other lesion',
                   'Lung tumor','Pneumonia','Tuberculosis','Other diseases','COPD','No finding']
        
        self.class_map = {name: idx for idx, name in enumerate(classes)}
        
        print("Classes:", self.class_map)
        return ann_train, ann_test, img_train, img_test
    
    def dicom_to_png(self, dicom_path):
        try:
            ds = pydicom.dcmread(dicom_path)
            img = ds.pixel_array

            # Corrigir inversão (Photometric Interpretation)
            if hasattr(ds, 'PhotometricInterpretation'):
                if ds.PhotometricInterpretation == "MONOCHROME1":
                    img = img.max() - img  # inverte contraste

            # Ajustar valores se necessário
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * ds.RescaleSlope + ds.RescaleIntercept

            # Normalizar para 0–255
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype('uint8')

            return img, img.shape[1], img.shape[0]

        except Exception as e:
            print(f"erro ao converter {dicom_path}: {e}")
            return None, 0, 0
    
    def find_dicom(self, image_id, folder):
        for ext in ['.dicom']:
            path = os.path.join(folder, f"{image_id}{ext}")
            if os.path.exists(path):
                return path
        return None
    
    def convert_bbox(self, x_min, y_min, x_max, y_max, img_w, img_h):
        #normalizar
        x_center = (x_min + x_max) / 2 / img_w
        y_center = (y_min + y_max) / 2 / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h
        
        
        #x_center = max(0, min(1, x_center))
        #y_center = max(0, min(1, y_center))
        #width = max(0, min(1, width))
        #height = max(0, min(1, height))
        
        return x_center, y_center, width, height
    
    def process_images(self, annotations, image_labels, dicom_folder, 
                      img_output_dir, label_output_dir, dataset_name):
        
        print(f"\nprocessando {dataset_name}...")
        
        image_ids = image_labels['image_id'].unique()
        processed = 0
        
        for img_id in image_ids:

            # Encontrar arquivo DICOM
            dicom_path = self.find_dicom(img_id, dicom_folder)
            
            if not dicom_path:
                print(f"DICOM não encontrado: {img_id}")
                continue
            # Converter para PNG
            image, width, height = self.dicom_to_png(dicom_path)
            
            if image is None:
                continue

            # salvar imagem
            img_path = os.path.join(img_output_dir, f"{img_id}.png")
            cv2.imwrite(img_path, image)

            # salvar label
            label_path = os.path.join(label_output_dir, f"{img_id}.txt")
            
            # busca annotations desta imagem
            img_annotations = annotations[annotations['image_id'] == img_id]
            
            # escrever arquivo de label
            with open(label_path, 'w') as f:
                for _, ann in img_annotations.iterrows():
                    class_name = ann['class_name']
                    if class_name in self.class_map:
                        class_id = self.class_map[class_name]
                        
                        # Converter bbox
                        xc, yc, w, h = self.convert_bbox(
                            ann['x_min'], ann['y_min'], 
                            ann['x_max'], ann['y_max'],
                            width, height
                        )
                        
                        # formato yolo
                        f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            
            processed += 1
            
        print(f"processadas {processed} imagens...")
        
        print(f"{dataset_name}: {processed} imagens processadas")
        return processed
    
    def create_data_yaml(self):
        classes = list(self.class_map.keys())
        
        yaml_content = f"""path: {os.path.abspath(self.output_dir)}
train: images/train
val: images/val
test: images/test
channels: 1

nc: {len(classes)}
names: {classes}
"""
        with open(os.path.join(self.output_dir, 'data.yaml'), 'w') as f:
            f.write(yaml_content)
        print("arquivo data.yaml criado")
    
    def split_train_val(self, split_ratio=0.2):

        train_img_dir = os.path.join(self.output_dir, 'images/train')
        train_label_dir = os.path.join(self.output_dir, 'labels/train')
        
        images = [f for f in os.listdir(train_img_dir) if f.endswith('.png')]
        
        
        # calculo quantas imagens para validação
        n_val = int(len(images) * split_ratio)
        val_images = images[:n_val]
        
        # mover para validação
        for img_file in val_images:
            # Mover imagem
            src_img = os.path.join(train_img_dir, img_file)
            dst_img = os.path.join(self.output_dir, 'images/val', img_file)
            shutil.move(src_img, dst_img)
            
            # Mover label
            label_file = img_file.replace('.png', '.txt')
            src_label = os.path.join(train_label_dir, label_file)
            dst_label = os.path.join(self.output_dir, 'labels/val', label_file)
            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)
        
        print(f"split realizado: {len(images)-n_val} treino, {n_val} validação")
    
    def run(self):
        print("Iniciando conversão DICOM para YOLO")
        
        # Carregar CSVs
        ann_train, ann_test, img_train, img_test = self.load_csvs()
       
        # Processar treino
        self.process_images(
            ann_train, img_train, self.dicom_train_path,
            os.path.join(self.output_dir, 'images/train'),
            os.path.join(self.output_dir, 'labels/train'),
            'treino'
        )
       
        # Processar teste
        self.process_images(
            ann_test, img_test, self.dicom_test_path,
            os.path.join(self.output_dir, 'images/test'),
            os.path.join(self.output_dir, 'labels/test'),
            'teste'
        )
      
        #Criar split treino/validação
        self.split_train_val()
        
        # Criar arquivo de configuração
        self.create_data_yaml()
        
        print("Imagens convertidas!")

if __name__ == "__main__":
    path = input("Digite o caminho do diretório onde está o arquivo Physionet: ")
    path = path if path.endswith('/') else path + '/'
    out = input("Digite o nome do diretório de saída (será criado se não existir): ")
    converter = DICOMToYOLO(
        annotations_train= path + "/physionet.org/files/vindr-cxr/1.0.0/annotations/annotations_train.csv",
        annotations_test= path + "/physionet.org/files/vindr-cxr/1.0.0/annotations/annotations_test.csv",
        image_labels_train= path +"/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_train.csv",
        image_labels_test= path +"/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_test.csv",
        dicom_train_path= path +"/physionet.org/files/vindr-cxr/1.0.0/train",
        dicom_test_path= path + "/physionet.org/files/vindr-cxr/1.0.0/test",
        output_dir= out
    )
    converter.run()