## Extração de características por meio de Redes Neurais com o Keras

### Passos iniciais

É necessário ter o TensorFlow(https://www.tensorflow.org/install/) instalado e também a biblioteca do Keras (https://keras.io/#installation).

### Preparando o conjunto de dados

Organize o conjunto de imagens separados por classe em pastas, como nesse exemplo:

	/diretorio -> /class1
	           -> /class2
	           -> /class3

### Funcionamento do Código

O primeiro argumento que se deve alterar é o img_size (linha 45), responsável por redimensionar as imagens do conjunto de dados conforme a arquitetura utilizada, segue o img_size para cada arquitetura:

Inception V3: 299

Inception ResNet V2: 299

NASNet-Large: 331

ResNet 50: 224

Após isso pode-se alterar o nome da pasta do conjunto de dados, por default está apenas 'diretorio' (linha 48).

O código implementa a extração de características com 4 arquiteturas diferentes, todas estão comentadas no código e apenas é possivel utilizar uma por vez enquanto as outras ficam comentadas. Após escolher uma arquitetura basta descomentar as linhas de cada uma. (É possivel realizar a extração já com as 4 redes de uma vez mudando apenas o nome de algumas variáveis para não ter conflito, mas não é recomendado porque somente uma arquitetura já aloca uma quantidade imensa de memória)

O processo pode ser um pouco demorado dependendo da máquina, recomenda-se o uso de uma boa GPU.

Para executar basta "python extract_features_nets_keras.py" no terminal, e ao final o programa salvará um arquivo .csv com todas características extraidas pela rede escolhida. Segue o número de features que cada rede é capaz de extrair(os arquivos ficam bem pesados):

Inception V3: 2048

Inception ResNet V2: 1536

NASNet-Large: 4032

ResNet 50: 2048