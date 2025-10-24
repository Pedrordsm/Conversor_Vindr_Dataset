import os
import pydicom
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
                   'Enlarged PA','ILD','Infiltration','Lung cavity','Lung cyst','Lung Opacity','Mediastinal shift',
                   'Nodule/Mass','Pulmonary fibrosis','Pneumothorax','Pleural thickening','Pleural effusion','Rib fracture','Other lesion']
        
        self.class_map = {name: idx for idx, name in enumerate(classes)}
        
        print("Classes:", self.class_map)
        return ann_train, ann_test, img_train, img_test
    
    def get_dicom_info(self, dicom_path):
        """Obtém informações do DICOM sem converter para PNG"""
        try:
            ds = pydicom.dcmread(dicom_path)
            width = ds.Columns
            height = ds.Rows
            
            return width, height

        except Exception as e:
            print(f"Erro ao ler DICOM {dicom_path}: {e}")
            return 0, 0
    
    def find_dicom(self, image_id, folder):
        for ext in ['.dicom']:
            path = os.path.join(folder, f"{image_id}{ext}")
            if os.path.exists(path):
                return path
        return None
    
    def copy_dicom_file(self, dicom_path, output_path):
        """Copia o arquivo DICOM para o diretório de destino"""
        try:
            shutil.copy2(dicom_path, output_path)
            return True
        except Exception as e:
            print(f"Erro ao copiar DICOM: {e}")
            return False
    
    def convert_bbox(self, x_min, y_min, x_max, y_max, img_w, img_h):
        #normalizar
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
            # Encontrar arquivo DICOM
            dicom_path = self.find_dicom(img_id, dicom_folder)
            
            if not dicom_path:
                print(f"DICOM não encontrado: {img_id}")
                continue
            
            # Obter dimensões do DICOM
            width, height = self.get_dicom_info(dicom_path)
            
            if width == 0 or height == 0:
                continue

            # Copiar arquivo DICOM para diretório de destino
            dicom_output_path = os.path.join(img_output_dir, f"{img_id}.dicom")
            if not self.copy_dicom_file(dicom_path, dicom_output_path):
                continue

            # Salvar label
            label_path = os.path.join(label_output_dir, f"{img_id}.txt")
            
            # Buscar annotations desta imagem
            img_annotations = annotations[annotations['image_id'] == img_id]
            
            # Escrever arquivo de label
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
                        
                        # Formato YOLO
                        f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            
            processed += 1
            
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
        print("Arquivo data.yaml criado")
    
    def split_train_val(self, split_ratio=0.2):
        train_img_dir = os.path.join(self.output_dir, 'images/train')
        train_label_dir = os.path.join(self.output_dir, 'labels/train')
        
        images = [f for f in os.listdir(train_img_dir) if f.endswith('.dicom')]
        
        # Cálculo quantas imagens para validação
        n_val = int(len(images) * split_ratio)
        val_images = images[:n_val]
        
        # Mover para validação
        for img_file in val_images:
            # Mover imagem
            src_img = os.path.join(train_img_dir, img_file)
            dst_img = os.path.join(self.output_dir, 'images/val', img_file)
            shutil.move(src_img, dst_img)
            
            # Mover label
            label_file = img_file.replace('.dicom', '.txt')
            src_label = os.path.join(train_label_dir, label_file)
            dst_label = os.path.join(self.output_dir, 'labels/val', label_file)
            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)
        
        print(f"Split realizado: {len(images)-n_val} treino, {n_val} validação")
    
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
      
        # Criar split treino/validação
        self.split_train_val()
        
        # Criar arquivo de configuração
        self.create_data_yaml()
        
        print("Conversão concluída!")

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