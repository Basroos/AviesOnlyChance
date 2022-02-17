import enum
import re
import random
import numpy as np
from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model

dataPath = 'train.to'
dataPath2 = 'train.from'

with open(dataPath, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

with open(dataPath2, 'r', encoding='utf-8') as f:
    lines2 = f.read().split('\n')

lines = [re.sub(r"\[\w+\]",'hi',line) for line in lines]
lines = [" ".join(re.findall(r"\w+",line)) for line in lines]
lines2 = [re.sub(r"\[\w+\]",'',line) for line in lines2]
lines2 = [" ".join(re.findall(r"\w+",line)) for line in lines2]

pairs = list(zip(lines,lines2))
#random.shuffle(pairs)

inputDocs = []
targetDocs = []
inputTokens = set()
targetTokens = set()

for line in pairs[:400]:
    inputDoc, targetDoc = line[0], line[1]
    inputDocs.append(inputDoc)
    targetDoc = " ".join(re.findall(r"[\w']+|[^\s\w]", targetDoc))
    targetDoc = '<START> ' + targetDoc + ' <END>'
    targetDocs.append(targetDoc)

    for token in re.findall(r"[\w']+|[^\s\w]", inputDoc):
        if token not in inputTokens:
            inputTokens.add(token)
    for token in targetDoc.split():
        if token not in targetTokens:
            targetTokens.add(token)

inputTokens = sorted(list(inputTokens))
targetTokens  = sorted(list(targetTokens))
numEncoderTokens = len(inputTokens)
numDecoderTokens = len(targetTokens)

inputFeaturesDict = dict(
    [(token, i) for i, token in enumerate(inputTokens)])
targetFeaturesDict = dict(
    [(token, i) for i, token in enumerate(targetTokens)])

reverseInputFeaturesDict = dict(
    (i, token) for token, i in inputFeaturesDict.items())
reverseTargetFeaturesDict = dict(
    (i, token) for token, i in targetFeaturesDict.items())

maxEncoderSeqLength = max([len(re.findall(r"[\w']+|[^\s\w]", inputDoc)) for inputDoc in inputDocs])
maxDecoderSeqLength = max([len(re.findall(r"[\w']+|[^\s\w]", targetDoc)) for targetDoc in targetDocs])

encoderInputData = np.zeros(
    (len(inputDocs), maxEncoderSeqLength, numEncoderTokens), dtype='float32')
decoderInputData = np.zeros(
    (len(inputDocs), maxDecoderSeqLength, numDecoderTokens), dtype='float32')
decoderTargetData = np.zeros(
    (len(inputDocs), maxDecoderSeqLength, numDecoderTokens), dtype='float32')

for line, (inputDoc, targetDoc) in enumerate(zip(inputDocs, targetDocs)):
    for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", inputDoc)):
        encoderInputData[line, timestep, inputFeaturesDict[token]] = 1.
    for timestep, token in enumerate(targetDoc.split()):
        decoderInputData[line, timestep, targetFeaturesDict[token]] = 1.
        if timestep > 0:
            decoderTargetData[line, timestep-1, targetFeaturesDict[token]] = 1.


dimensionality = 256
batchSize = 10
epochs = 600

# https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/

encoderInputs = Input(shape=(None, numEncoderTokens))
encoderLstm = LSTM(dimensionality, return_state=True)
encoderOutputs, stateHidden, stateCell = encoderLstm(encoderInputs)
encoderStates = [stateHidden, stateCell]

decoderInputs = Input(shape=(None, numDecoderTokens))
decoderLstm = LSTM(dimensionality, return_sequences=True, return_state=True)
decoderOutputs, decoderStateHidden, decoderStateCell = decoderLstm(decoderInputs, initial_state=encoderStates)
decoderDense = Dense(numDecoderTokens, activation='softmax')
decoderOutputs = decoderDense(decoderOutputs)

trainingModel = Model([encoderInputs, decoderInputs], decoderOutputs)

trainingModel.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')

trainingModel.fit([encoderInputData, decoderInputData],
                  decoderTargetData,
                  batch_size = batchSize,
                  epochs=epochs,
                  validation_split = 0.2)
trainingModel.save('Models/trainingModel.h5')

trainingModel = load_model('Models/trainingModel.h5')

encoderInputs = trainingModel.input[0]
encoderOutputs, stateHEnc, stateCEnc = trainingModel.layers[2].output
encoderStates = [stateHEnc, stateCEnc]
encoderModel = Model(encoderInputs, encoderStates)

latentDim = 256
decoderStateInputHidden = Input(shape=(latentDim,))
decoderStateInputCell = Input(shape=(latentDim,))
decoderStatesInputs = [decoderStateInputHidden, decoderStateInputCell]

decoderOutputs, stateHidden, stateCell = decoderLstm(decoderInputs, initial_state=decoderStatesInputs)
decoderStates = [stateHidden, stateCell]
decoderOutputs = decoderDense(decoderOutputs)

decoderModel = Model([decoderInputs] + decoderStatesInputs, [decoderOutputs] + decoderStates)

def decodeResponse(testInput):
    statesValue = encoderModel.predict(testInput)
    targetSeq = np.zeros((1, 1, numDecoderTokens))
    targetSeq[0, 0, targetFeaturesDict['<START>']] = 1.
    decodedSentence = ''

    stopCondition = False

    while not stopCondition:
        outputTokens, hiddenState, cellState = decoderModel.predict([targetSeq] + statesValue)

        sampledTokenIndex = np.argmax(outputTokens[0, -1, :])
        sampledToken = reverseTargetFeaturesDict[sampledTokenIndex]
        decodedSentence += " " + sampledToken

        if sampledToken == '<END>' or len(decodedSentence) > maxDecoderSeqLength:
            stopCondition = True

        targetSeq = np.zeros((1, 1, numDecoderTokens))
        targetSeq[0, 0, sampledTokenIndex] = 1.
        statesValue = [hiddenState, cellState]
        return decodedSentence

class ChatBot:
    negativeResponses = ("no", "nope", "nah", "naw", "not a chance", "sorry")
    exitCommands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")

    def startChat(self):
        userResponse = input('hi, would you like to chat with me?\n')
        if userResponse in self.negativeResponses:
            print('ok, fuck you then')
            return
        self.chat(userResponse)
    
    def chat(self, reply):
        while not self.makeExit(reply):
            reply = input(self.generateResponse(reply) + "\n")

    def stringToMatrix(self, userInput):
        tokens = re.findall(r"[\w']+|[^\s\w]", userInput)
        userInputMatrix = np.zeros((1, maxEncoderSeqLength, numEncoderTokens), dtype='float32')
        for timestep, token in enumerate(tokens):
            if token in inputFeaturesDict:
                userInputMatrix[0, timestep, inputFeaturesDict[token]] = 1.
        return userInputMatrix

    def generateResponse(self, userInput):
        inputMatrix = self.stringToMatrix(userInput)
        chatBotResponse = decodeResponse(inputMatrix)
        chatBotResponse = chatBotResponse.replace("<START>", '')
        chatBotResponse = chatBotResponse.replace("<END>", '')
        return chatBotResponse

    def makeExit(self, reply):
        for exitCommand in self.exitCommands:
            if exitCommand in reply:
                print("ok, bye")
                return True
        return False
        
chatbot = ChatBot()
