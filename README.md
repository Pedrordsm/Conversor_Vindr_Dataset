# Conversor do Dataset Vindr-CXR para YOLOvX

Este repositório converte imagens **DICOM** para o formato **PNG** e organiza o dataset em 3 subpastas (**`train`**, **`val`**, **`test`**) prontas para o treinamento de modelos **YOLO** (You Only Look Once).

### Autor

- Pedro Renã da Silva Moreira: [@Pedrordsm](https://github.com/Pedrordsm)

# Instalação e Execução
## Pré-requisitos
Antes de começar, certifique-se de ter instalado:

* **Python 3.x:** (Recomendado 3.8+)
* O **Dataset Vindr-CXR** (baixado do PhysioNet ou fonte original)
* Um compilador Python (que geralmente vem com a instalação do Python).

Para instalar o programa em seu computador, siga os passos a seguir:
1. **Clone o repositório em seu Desktop e salve-o no local desejado:**
```bash
git clone https://github.com/Pedrordsm/Conversor_Vindr_Dataset.git
```
2. **Abra seu terminal, escreva ```cd``` e cole o caminho no qual o repositório está salvo, semelhante ao exemplo:**

```
cd C:\Users\NomedoUsuario\Documents\Conversor_Vindr_Dataset
```

## Criar e utilizar um ambiente virtual
Para evitar conflitos com outros projetos Python, use um ambiente virtual (venv).

### Criação do ambiente virtual
```bash
# No Windows
py -m venv .venv

# No Linux/WSL
python3 -m venv .venv
```
### Ativação do ambiente virtual

```bash
# No Windows (PowerShell)
.\.venv\Scripts\activate

# No Linux/WSL
source .venv/bin/activate
```

3. **Instale os requirements**
```bash
pip install -r requirements.txt
```
4. **Execute o script:**

```bash
python converter.py
```

# Funcionamento do programa
Ao executar o programa, o usuário deverá passar os caminhos de entrada e o nome da pasta de saída via terminal: 

```
Digite o caminho do diretório onde está o arquivo Physionet: (pasta onde o arquivo do physionet.org está)

Digite o nome do diretório de saída (será criado se não existir): exemplosaida
```

Após passar corretamente o processo começará a ser executado e terminará sozinho:

```
Pastas criadas
Iniciando conversão DICOM para YOLO
...
...
...
Imagens convertidas!

```
