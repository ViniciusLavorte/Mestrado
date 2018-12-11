import csv
from time import sleep

class convert(object):

    content = []
    name = ''

    def __init__(self):
        self.csvInput()
        self.arffOutput()
        print 'Terminado.'

    #abre o arquivo csv
    def csvInput(self):

        user = raw_input('Entre com o nome do arquivo CSV (ex: test.csv): ')

        #remove o .csv
        if user.endswith('.csv') == True:
            self.name = user.replace('.csv', '')
            
        print 'Abrindo o CSV.'     
        try:
            with open(user, 'rb') as csvfile:
               lines = csv.reader(csvfile, delimiter = ',')
               for row in lines:
                   self.content.append(row)
            csvfile.close()
            sleep(2)
            
        #exception caso o usuario tente entrar com um arquivo nao existente
        except IOError:
            sleep(2)
            print 'Arquivo nao encontrado.\n'
            self.csvInput()
            
    #exportando o ARFF
    def arffOutput(self):
        print 'Convertendo para arquivo ARFF.\n'
        title = str(self.name) + '.arff'
        new_file = open(title, 'w')

        #escrevendo as relacoes
        new_file.write('@relation ' + str(self.name)+ '\n\n')

	aux = 0
        #pegando o tipo do atributo de entrada
        for i in range(len(self.content[0])-1):
            #alterar aqui para csv's menores
            #attribute_type = raw_input('Is the type of ' + str(self.content[0][i]) + ' numeric or nominal? ') 

            attribute_type = 'numeric'
            #new_file.write('@attribute ' + str(self.content[0][i]) + ' ' + str(attribute_type) + '\n')
	    aux = aux + 1
	    new_file.write('@attribute ' + str(aux) + ' ' + str(attribute_type) + '\n')

        #criando uma lista para os atributos classes
        last = len(self.content[0])
        class_items = []
        for i in range(len(self.content)):
            name = self.content[i][last-1]
            if name not in class_items:
                class_items.append(self.content[i][last-1])
            else:
                pass  
    
        string = '{' + ','.join(sorted(class_items)) + '}'
        new_file.write('@attribute class' + ' ' + str(string) + '\n')

        #escreve os novos dados
        new_file.write('\n@data\n')

        for row in self.content:
            new_file.write(','.join(row) + '\n')

        #fecha o arquivo
        new_file.close()
        sleep(2)

    
#####    
run = convert()
