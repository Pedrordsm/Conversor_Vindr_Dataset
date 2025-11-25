import os
import pydicom
import cv2
import pandas as pd
import shutil
import argparse

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
        
        self.create_folders()
        
    def create_folders(self):
        folders = ['images/train', 'images/val',
                  'labels/train', 'labels/val']
        
        for folder in folders:
            os.makedirs(os.path.join(self.output_dir, folder), exist_ok=True)
        print("Pastas criadas!")
    
    def load_csvs(self):
        #carregar annotations
        ann_train = pd.read_csv(self.annotations_train)
        ann_test = pd.read_csv(self.annotations_test)
        
        #carregar image labels
        img_train = pd.read_csv(self.image_labels_train)
        img_test = pd.read_csv(self.image_labels_test)
        
        #mapeamento de classes
        classes = ['Aortic enlargement','Atelectasis','Cardiomegaly','Calcification','Clavicle fracture','Consolidation','Edema','Emphysema',
                   'Enlarged PA','ILD','Infiltration','Lung cavity','Lung cyst','Lung Opacity','Mediastinal shift',
                   'Nodule/Mass','Pulmonary fibrosis','Pneumothorax','Pleural thickening','Pleural effusion','Rib fracture','Other lesion']
        
        self.class_map = {name: idx for idx, name in enumerate(classes)}
        
        print("Classes:", self.class_map)
        return ann_train, ann_test, img_train, img_test
    
    def dicom_to_png(self, dicom_path):
        try:
            ds = pydicom.dcmread(dicom_path)
            img = ds.pixel_array

            #corrigir inversão
            if hasattr(ds, 'PhotometricInterpretation'):
                if ds.PhotometricInterpretation == "MONOCHROME1":
                    img = img.max() - img  # inverte contraste

            #ajustar valores se necessário
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * ds.RescaleSlope + ds.RescaleIntercept

            #normalizar para 0-255
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype('uint8')

            return img, img.shape[1], img.shape[0]

        except Exception as e:
            print(f"Erro ao converter {dicom_path}: {e}")
            return None, 0, 0
    
    def find_dicom(self, image_id, folder):
        for ext in ['.dicom']:
            path = os.path.join(folder, f"{image_id}{ext}")
            if os.path.exists(path):
                return path
        return None
    
    def convert_bbox(self, x_min, y_min, x_max, y_max, img_w, img_h):
        #normalizar coordenadas para formato YOLO
        x_center = (x_min + x_max) / 2 / img_w
        y_center = (y_min + y_max) / 2 / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h
        
        return x_center, y_center, width, height
    
    def process_images(self, annotations, image_labels, dicom_folder, 
                      img_output_dir, label_output_dir, dataset_name):
        
        print(f"\nProcessando {dataset_name}...")
        
        image_ids = image_labels['image_id'].unique()
        processed = 0
        
        for img_id in image_ids:
            #encontrar arquivo DICOM
            dicom_path = self.find_dicom(img_id, dicom_folder)
            
            if not dicom_path:
                print(f"DICOM não encontrado: {img_id}")
                continue
            
            #converter para PNG
            image, width, height = self.dicom_to_png(dicom_path)
            
            if image is None:
                continue

            #salvar imagem PNG
            img_path = os.path.join(img_output_dir, f"{img_id}.png")
            cv2.imwrite(img_path, image)

            #salvar label
            label_path = os.path.join(label_output_dir, f"{img_id}.txt")
            
            #buscar annotations desta imagem
            img_annotations = annotations[annotations['image_id'] == img_id]
            
            #escrever arquivo de label
            with open(label_path, 'w') as f:
                for _, ann in img_annotations.iterrows():
                    class_name = ann['class_name']
                    if class_name in self.class_map:
                        class_id = self.class_map[class_name]
                        
                        #converter bbox
                        xc, yc, w, h = self.convert_bbox(
                            ann['x_min'], ann['y_min'], 
                            ann['x_max'], ann['y_max'],
                            width, height
                        )
                        
                        #formato YOLO classe x-center y-center width height
                        f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            
            processed += 1
            
        print(f"{dataset_name}: {processed} imagens processadas")
        return processed
    
    def create_data_yaml(self):
        classes = list(self.class_map.keys())
        
        yaml_content = f"""path: {os.path.abspath(self.output_dir)}
train: images/train
val: images/val
channels: 1

nc: {len(classes)}
names: {classes}
"""
        with open(os.path.join(self.output_dir, 'data.yaml'), 'w') as f:
            f.write(yaml_content)
        print("Arquivo data.yaml criado")
    
    def run(self):
        print("Iniciando conversão DICOM para YOLO")
        
        #carregar CSVs
        ann_train, ann_test, img_train, img_test = self.load_csvs()
       
        #processar treino
        self.process_images(
            ann_train, img_train, self.dicom_train_path,
            os.path.join(self.output_dir, 'images/train'),
            os.path.join(self.output_dir, 'labels/train'),
            'treino'
        )
       
        #processar validação
        self.process_images(
            ann_test, img_test, self.dicom_test_path,
            os.path.join(self.output_dir, 'images/val'),
            os.path.join(self.output_dir, 'labels/val'),
            'validação'
        )
        
        #criar arquivo de configuração
        self.create_data_yaml()
        
        print("Conversão concluída!")

def main():
    parser = argparse.ArgumentParser(description='Converter dataset DICOM para formato YOLO')
    parser.add_argument('--physionet_path', type=str, required=True,
                       help='Caminho do diretório onde está o arquivo Physionet')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Nome do diretório de saída (será criado se não existir)')
    
    args = parser.parse_args()
    
    #garantir que o caminho termina com /
    physionet_path = args.physionet_path if args.physionet_path.endswith('/') else args.physionet_path + '/'
    
    converter = DICOMToYOLO(
        annotations_train= physionet_path + "physionet.org/files/vindr-cxr/1.0.0/annotations/annotations_train.csv",
        annotations_test= physionet_path + "physionet.org/files/vindr-cxr/1.0.0/annotations/annotations_test.csv",
        image_labels_train= physionet_path + "physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_train.csv",
        image_labels_test= physionet_path + "physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_test.csv",
        dicom_train_path= physionet_path + "physionet.org/files/vindr-cxr/1.0.0/train",
        dicom_test_path= physionet_path + "physionet.org/files/vindr-cxr/1.0.0/test",
        output_dir= args.output_dir
    )
    converter.run()

if __name__ == "__main__":
    main()