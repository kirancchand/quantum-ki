from django.http import JsonResponse
from django.shortcuts import render
import json
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, IBMQ,BasicAer
from qiskit.tools.monitor import job_monitor
import numpy as np
from qiskit.utils import QuantumInstance
# from qiskit.aqua import QuantumInstance, aqua_globals
# from qiskit.aqua.operators import SecondOrderExpansion

# from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.circuit.library import ZFeatureMap
from qiskit.aqua.components.multiclass_extensions import (ErrorCorrectingCode,AllPairs,OneAgainstRest)
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.utils import get_feature_dimension
from qiskit.algorithms import Shor
from qiskit.ml.datasets import breast_cancer


IBMQ.enable_account('7cca07ecd68c807c58ffd0f89251e86b50b8b9da289b6e8eaa5925a8842429336459f4ecc4349e937fac10af116ad81e3305ddad35022c376d419a78fc1e0818')
provider = IBMQ.get_provider(hub='ibm-q')

def index(request):
    return render(request, 'index.html')
def home(request):
    return render(request, 'home.html')
def requesthandler(request,id):
    if int(id)==1: 
        return render(request, 'randomnumbergenerator.html')
    elif int(id)==2: 
        return render(request, 'smokerprediction.html')
    elif int(id)==3: 
        return render(request, 'shorfactor.html')
    
def randomnumbergenerator(request):
    return render(request, 'randomnumbergenerator.html')
def random(request):

    print(request.body)

    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode);

    device = body['device']

    min = int(body['min'])
    max = int(body['max'])

    backend = provider.get_backend(device)

    if device == "ibmq_qasm_simulator":
        num_q = 32
    else:
        num_q = 5

    q = QuantumRegister(num_q, 'q')
    c = ClassicalRegister(num_q, 'c')

    circuit = QuantumCircuit(q, c)
    circuit.h(q)  # Applies hadamard gate to all qubits
    circuit.measure(q, c)  # Measures all qubits


    job = execute(circuit, backend, shots=1)

    print('Executing Job...\n')
    job_monitor(job)
    counts = job.result().get_counts()

    print('RESULT: ', counts, '\n')
    print('Press any key to close')

    result = int(counts.most_frequent(), 2)
    result1 = min + result % (max+1 - min)

    print(result1)

    response = JsonResponse({'result': result1})
    return response

def smokerprediction(request):

    # print('Quantum SVM')
    # print('-----------\n')

    # shots = 8192 # Number of times the job will be run on the quantum device 

    # training_data = {'A': np.asarray([[0.324],[0.565],[0.231],[0.756],[0.324],[0.534],[0.132],[0.344]]),'B': np.asarray([[1.324],[1.565],[1.231],[1.756],[1.324],[1.534],[1.132],[1.344]])}
    # testing_data = {'A': np.asarray([[0.024],[0.456],[0.065],[0.044],[0.324]]),'B': np.asarray([[1.777],[1.341],[1.514],[1.204],[1.135]])}

    # backend = provider.get_backend('ibmq_qasm_simulator') # Specifying Quantum device

    # num_qubits = 1
    # # num_qubits = 2
    # # feature_map = SecondOrderExpansion(feature_dimension=num_qubits,depth=2,entanglement='full')
    # # feature_map = ZZFeatureMap(feature_dimension=num_qubits,reps=2)
    # feature_map = ZFeatureMap(feature_dimension=num_qubits,reps=1)

    # svm = QSVM(feature_map, training_data,testing_data) # Creation of QSVM

    # quantum_instance = QuantumInstance(backend,shots=shots,skip_qobj_validation=False)

    # print('Running....\n')

    # result = svm.run(quantum_instance) # Running the QSVM and getting the accuracy
    # try:
    #     job_status = job.status()  # Query the backend server for job status.
    #     if job_status is JobStatus.RUNNING:
    #         print("The job is still running")
    # except IBMQJobApiError as ex:
    #     print("Something wrong happened!: {}".format(ex))
    # print('Result....\n')
    # print(result)
    # data = np.array([[1.453],[1.023],[0.135],[0.266]]) #Unlabelled data

    # prediction = svm.predict(data,quantum_instance) # Predict using unlabelled data 

    # print('Prediction of Smoker or Non-Smoker based upon gene expression of CDKN2A\n')
    # print('Accuracy: ' , result['testing_accuracy'],'\n')
    # print('Prediction from input data where 0 = Non-Smoker and 1 = Smoker\n')
    # print(prediction)
    # return prediction
    feature_dim = 2
    sample_total, training_input, test_input, class_labels = breast_cancer(training_size=20,test_size=10,n=feature_dim,plot_data=True)
    feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement='linear')
    qsvm = QSVM(feature_map, training_input, test_input)

    backend = BasicAer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed)

    result = qsvm.run(quantum_instance)
    print(result)

    print(f'Testing success ratio: {result["testing_accuracy"]}')
def factor(request):

    print(request.body)

    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode);

    device = body['device']
    number = int(body['number'])

    backend = provider.get_backend(device)

    factors = Shor(QuantumInstance(backend, shots=1, skip_qobj_validation=False)) #Function to run Shor's algorithm where 21 is the integer to be factored

    result_dict = factors.factor(N=number, a=2)
    result = result_dict.factors

    response = JsonResponse({'result': str(result)})
    return response
