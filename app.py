import streamlit as st
from extract_mfcc import save_mfcc
import numpy as np
import tensorflow as tf
st.title("Speech Recognition App")
st.markdown("Welcome to the Speech Recognition App. This app will convert your speech to text. Please click on the upload button below to upload your audio file. The app will then convert your speech to text and display it below.")

SAVED_MODEL_PATH = 'model.h5'
uploaded_file = st.file_uploader("Upload your audio file", type=["wav","mp3"])
print(uploaded_file)
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.write("Converting your speech to Arabic Word...")
    mfcc_l = save_mfcc(uploaded_file)
    print('mfcc',mfcc_l)

    class _Keyword_Spotting_Service:
        """Singleton class for keyword spotting inference with trained models.

        :param model: Trained model
        """
        
        model = None
        _mapping = [
        "بِسۡمِ [Bis'mi]",
        "لِلَّهِ [Al-lahi]",
        "ٱلرَّحۡمَٰنِ[Al-rahmaani]",
        "ٱلرَّحِيمِ [Al-raheemi]",
        "ٱلۡحَمۡدُ [Alhamdu]",
        "لِلَّهِ [lillaahi]",
        "رَبِّ [Rabbil]",
        "ٱلۡعَٰلَمِينَ [aalameen]",
        "ٱلرَّحۡمَٰنِ  [Ar-Rahmaan]",
        "ٱلرَّحِيمِ [Ar-Raheem]",
        "مَٰلِكِ  [Maaliki]",
        "يَوۡمِ  [Yumid]",
        "ٱلدِّينِ [Diin]",
        "إِيَّاكَ  [Iyyaka]",
        "نَعۡبُدُ  [Na'abudu]",
        "وَإِيَّاكَ  [Iyyaka]",
        "نَسۡتَعِينُ [Nasta'een]",
        "ٱهۡدِنَا  [Ihdinas]",
        "ٱلصِّرَٰطَ  [Siraatal]",
        "ٱلۡمُسۡتَقِيمَ [Mustaqeem]",
        "صِرَٰطَ  [Siraatal]",
        "ٱلَّذِينَ  [Ladheena]",
        "أَنۡعَمۡتَ  [An'amta]",
        "عَلَيۡهِمۡ  [Alaihim]",
        "غَيۡرِ  [Ghayril]",
        "ٱلۡمَغۡضُوبِ  [Maghdubi]",
        "عَلَيۡهِمۡ [Alaihim]",
        "وَلَا ٱلضَّآلِّينَ [Wala al-dalina]"
        ]
        _instance = None


        def predict(self, file_path):
            """

            :param file_path (str): Path to audio file to predict
            :return predicted_keyword (str): Keyword predicted by the model
            """

            # extract MFCC
            MFCCs = np.array(mfcc_l)

            # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
            MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

            # get the predicted label
            predictions = self.model.predict(MFCCs) # a 2d array [[]]
            predicted_index = np.argmax(predictions)
            # index return the index which has highest score
            predicted_keyword = self._mapping[predicted_index]
            # print('prediction',predicted_index,predicted_keyword)
            return predicted_keyword




    def Keyword_Spotting_Service():
        """Factory function for Keyword_Spotting_Service class.

        :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
        """

        # ensure an instance is created only the first time the factory function is called
        if _Keyword_Spotting_Service._instance is None:
            _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
            _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
        return _Keyword_Spotting_Service._instance




    if __name__ == "__main__":

        # create 2 instances of the keyword spotting service
        kss = Keyword_Spotting_Service()
        kss1 = Keyword_Spotting_Service()

        # check that different instances of the keyword spotting service point back to the same object (singleton)
        assert kss is kss1

        # make a prediction
        keyword = kss.predict(uploaded_file)
        print(keyword)
        st.write("Your speech has been converted to the following Arabic word:")
        st.title(keyword)