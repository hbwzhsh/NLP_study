from tkinter import *
import tkinter.messagebox as messagebox

from toxic_comment_classification_CNN_GRU_Glove_predict import *

# sorts
class_names = ['toxic', 'severe_toxic', 'obscene',
               'threat', 'insult', 'identity_hate']


class Application(Frame):
    def __init__(self, ppl_process, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()
        self.ppl_process = ppl_process

    def createWidgets(self):
        self.textInput = Text(
            self, width=50, height=9, font=('Courier', '14', 'bold'))
        self.textInput.pack()
        self.alertButton = Button(
            self, text='Analyze',  font=('Courier', '16', 'bold'), command=self.process)
        self.alertButton.pack()

    def process(self):
        # 1.0: 第一行第一个字母
        text = self.textInput.get('1.0', END)
        if (text == "\n"):
            text = "Please input texts!"
            messagebox.showinfo('Analysis Results', text)
        else:
            text = [text.replace('\n', ' ')]  # 去掉\n,turn into array
            toxicity = self.ppl_process.predict(text)
            # prediction output
            prediction = ""
            for text, toxicity in zip(text, self.ppl_process.predict(text)):
                for index, value in zip(class_names, toxicity):
                    prediction += "Toxicity of %-15s: %.5f\n" % (index, value)
            # print(prediction)
            messagebox.showinfo('Analysis Results', prediction)


if __name__ == "__main__":
    print("Loding model...")
    # *号表示：以元祖形式导入preprocess和model
    ppl = PredictionPipeline(*load_pipeline(PREPROCESSOR_FILE,
                                            ARCHITECTURE_FILE,
                                            WEIGHTS_FILE))
    print("Complete loding model!")

    # run app
    app = Application(ppl)
    # set window title:
    app.master.title('Toxic Comment Classification')
    # process loop:
    app.mainloop()
