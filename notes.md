Multi-processing will likely require that all the models and set up is done before any threads are called, with any models or necessary info for
STT, TTS, or Face Verification passed as references to functions that create classes of the afformentioned functionalities, with the function passing
the model references and such to those classes. Those classes will solely be responsible for calls to any models, doing any processing, communicating
with database, etc. They will act independently within their thread, with the wrapping function being responsible for taking any output and processing
it into the buffer, from which the main thread will be able to read it.