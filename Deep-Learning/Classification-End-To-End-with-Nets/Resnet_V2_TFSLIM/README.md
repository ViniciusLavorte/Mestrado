## Inception ResNet V2

Esse é um código para realizar a extração de características e classificação (end-to-end) de um dado conjunto de imagens por meio de transferência de aprendizado.

É necessário ter o TensorFlow(https://www.tensorflow.org/install/) e a biblioteca TF-Slim(https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) instalados.

### Passos iniciais

Clone esse repositório e crie as seguintes pastas:

    /datasets
    /train_logs
    /eval_logs

Também é necessário baixar o checkpoint da rede pré treinada e colocar dentro do repositório.

Link para o Checkpoint da Inception ResNet V2: http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz

### Preparando o conjunto de dados

As imagens podem estar em tamanhos diferentes o próprio algoritmo irá fazer um pré processamento redimensionando e etc.

Dentro da pasta /dataset deverá ser colocado o conjunto de imagem, separando as classes por pasta, segue um exemplo:

/datasets -> /iris-dataset -> /iris-setosa
                           -> /iris-virginica
                           -> /iris-versicolor

Depois de colocar o conjunto de imagens dessa maneira, a primeira coisa a se fazer é criar os arquivos .tfrecords que são streams de imagens otimizadas para que a rede leia de maneira mais rápida o conjunto de imagens. 
Para isso basta rodar o arquivo "python /create_tfrecords-master/create_tfrecord.py".

Por default o arquivo "create_tfrecord.py" está configurado para separar o conjunto em 80% treino e 20% teste, é possível alterar na linha 12 do código, o atributo "validation_size". Como por ultimo eu trabalhei com o Mammoset, os arquivos também estão salvando com esse nome, mas basta alterar também na linha 21 o atributo "tfrecord_filename".

Após isso o algoritmo terá criado os labels e os arquivos .tfrecords na própria pasta /datasets. A cada run da rede é interessante apagar esses arquivos e criá-los novamente para ter uma variabilidade, a não ser que deseja fazer o mesmo experimento com redes diferentes.

### Treinando a rede

Temos o arquivo /transfer_learning-master/train.py que é responsável pelas tarefas de treino e validação da rede.

Ele possui alguns hiperparâmetros (epocas, batch_size, learning rate, etc) já definidos mas é possível alterar, basta olhar no código que eu deixei eles comentados já. É possivel também alterar os diretórios de logs, datasets e arquivo ckpt, basta alterar lá.

Um passo importante é alterar o parâmetro "num_classes" da linha 24, conforme o número de classes do seu experimento.

Após isso basta rodar o arquivo "python /transfer_learning-master/train.py". 

A cada época o código vai imprimindo no terminal as predições e acurácia atual da rede.

Esses códigos trabalham muito melhor com GPU, e dependendo do poder computacional que voce vai utilizar o batch_size deve ser mínimo. Em meu computador com uma GTX 1060 6GB consigo utilizar batch_size 16 para essa arquitetura. O batch_size basicamente é o número de amostras que o algoritmo pegará por iteração.

Ao final será criado arquivos logs no diretório /train_logs que podem ser abertos pelo TensorBoard para análise mais detalhada do aprendizado da rede, explico isso mais para frente.

### Testando a classificação da rede

Temos o arquivo /transfer_learning-master/eval.py que é responsável por testar o a classificação da rede.

Ele é mais simples de configurar, pode-se alterar os diretórios também do datasets, eval_logs e do train_logs.

Aqui também é importante alterar o batch_size conforme sua máquina. Em minha GTX 1060 6GB consigo utilizar batch_size 36 para os testes.

O número de épocas também pode ser alterado.

Para rodar o código basta "python /transfer_learning-master/eval.py" e ao final ele salvará logs na pasta /eval_logs.

### Visualizando os logs com o TensorBoard

É possível ter acesso a todos dados da rede ao longo de seu aprendizado e classificação, basta digitar esse comando no terminal:

tensorboard --logdir=run1:/Resnet_V2_TFSLIM/train_logs,run2:/Resnet_V2_TFSLIM/eval_logs

Ele combinará os dois logs juntos, é bem interessante. É possivel fazer só para o treinamento ou só classificação também.

Também é possivel ir visualizando no TensorBoard enquanto a rede ainda está sendo treinada, simultaneamente.