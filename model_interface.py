from tkinter import *
from PIL import Image,ImageTk
from tkinter import ttk
from tkinter import filedialog as fd
import cv2
import numpy as np
import time
import os
import webbrowser




project = Tk()
project.title("Landmark Detector")
project.geometry("360x550+400+100")
project.configure(bg="#484848")
project.resizable(width=False,height=False)

f3 = Frame(project,background="#484848")
f3.pack(side="top",anchor="n")
l1 = Label(master=f3,text="LANDMARK DETECTOR",font=("Times new roman",20,"bold"),bg="#484848",fg="#DAA520")
l1.pack(pady=1)

image = Image.open(r"C:\virtual enviroments\landmark-detection\project folder\intro_image.jpg")
photo = ImageTk.PhotoImage(image)

photo_label = Label(image = photo,bg="#DAA520",relief="flat",borderwidth=4)
photo_label.pack()


f1 = Frame(project)
f1.pack(side="left")

f2 = Frame(project)
f2.pack(side="right")



# and lay them out
top = Frame(project,background="#484848")
bottom = Frame(project)
top.pack(side=TOP)
bottom.pack(side=BOTTOM, fill=BOTH, expand=True)

# create the widgets for the top part of the GUI,
# and lay them out



def Gallery():
    filetypes = (
        ('media files', '*.jpg'),
        ('All files', '*.*')
    )

    filenames = fd.askopenfilenames(
        title='Open files',
        initialdir='/',
        filetypes=filetypes)


    try:
        # read input image
        image = cv2.imread(os.path.join(os.getcwd(),"detected images",filenames[0]))

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        # read class names from text file
        classes = None
        with open(r"C:\virtual enviroments\landmark-detection\project folder\classes.names", 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        # generate different colors for different classes 
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        # read pre-trained model and config file
        net = cv2.dnn.readNet(r"C:\virtual enviroments\landmark-detection\project folder\yolov4_custom_8000.weights", r"C:\virtual enviroments\landmark-detection\project folder\yolov4_custom.cfg")

        # create input blob 
        blob = cv2.dnn.blobFromImage(image, scale, (512,512), (0,0,0), True, crop=False)

        # set input blob for the network
        net.setInput(blob)




        # function to get the output layer names 
        # in the architecture
        def get_output_layers(net):
            
            layer_names = net.getLayerNames()
            
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            return output_layers

        # function to draw bounding box on the detected object with class name
        def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

            label = str(classes[class_id])

            color = (0,0,0)

            cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

            cv2.putText(img, label, (x-90,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)







        # run inference through the network
        # and gather predictions from output layers
        outs = net.forward(get_output_layers(net))

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # for each detetion from each output layer 
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])






        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # go through the detections remaining
        # after nms and draw bounding box
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            
            draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))


        # wait until any key is pressed
        cv2.waitKey()


        #save output image to disk
        cv2.imwrite(os.path.join(os.getcwd(),"detected images","object_detection"+str(time.time())+".jpg"), image)

        # release resources
        cv2.destroyAllWindows()

        project.withdraw()
        sub_project1 = Toplevel(project)
        sub_project1.geometry("800x505+200+100")
        sub_project1.resizable(width=False,height=False)
        sub_project1.title("landmark Detector")


        f4 = Frame(sub_project1,bg = "gray",height=600,relief="sunken",borderwidth=5)
        f4.pack(side=TOP,fill="x")
        


        
        file = os.listdir(os.path.join(os.getcwd(),"detected images"))[-1]
        print(os.path.join(os.getcwd(),"detected images",file))
        image3 = Image.open(os.path.join(os.getcwd(),"detected images",file))
        photo2 = image3.resize((600,400))
        photo2 = ImageTk.PhotoImage(photo2)
        photo_label2 = Label(f4,image = photo2,relief="flat")
        photo_label2.pack(pady=2)


        print(classes,class_ids[i])
        result = Label(f4,text="landmark detected"+" "+classes[class_ids[i]],font=("Calibri",15,"bold"),bg="gray",fg="#a2ff47")
        result.pack(side=TOP,pady=2)
        
        f5 = Frame(sub_project1,bg = "#484848",height=500)
        f5.pack(side=TOP,fill="x")

        def Search():
            webbrowser.open("https://en.wikipedia.org/wiki/"+classes[class_ids[i]])

        b3 = Button(f5,text = "SEARCH",relief="flat",width=9,height=5,font=("Calibri",20),bg="#DAA520",command=Search)
        b3.pack(side=LEFT,padx=50,pady=5)

        def go_back():
            sub_project1.destroy()
            project.deiconify()
                 
        b4 = Button(f5,text = "BACK",relief="flat",width=9,height=5,font=("Calibri",20),bg="#DAA520",command=go_back)
        b4.pack(side=RIGHT,padx=50,pady=5)
        sub_project1.mainloop()
    except Exception as e:
        pass

b2 = Button(project, text="Gallery",width=9, height=1,relief="flat", bg="#DAA520",font=("Calibri",20),command=Gallery)
    
    

def Camera():
    project.withdraw()
    sub_project1 = Toplevel(project)
    sub_project1.geometry("800x555+200+100")
    sub_project1.resizable(width=False,height=False)
    sub_project1.title("landmark Detector")

    
    f4 = Frame(sub_project1,bg = "gray",height=500,relief="sunken",borderwidth=5)
    f4.pack(side=TOP,fill="x")

    lmain = Label(f4)
    lmain.pack(anchor="n")
    cap = cv2.VideoCapture(0)
    def show_frame():
        _, frame = cap.read()
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        cv2image = cv2.resize(cv2image,(600,450))
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk #Shows frame for display 1
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)            
    show_frame()

    result = Label(f4,text="landmark detected",font=("Calibri",15,"bold"),bg="gray",fg="#a2ff47")
    result.pack(side=TOP,pady=2)
    
    f5 = Frame(sub_project1,bg = "#484848",height=300)
    f5.pack(side=TOP,fill="x")

    def Search():
        pass
        #webbrowser.open("https://en.wikipedia.org/wiki/"+classes[class_ids[i]])

    b3 = Button(f5,text = "SEARCH",relief="flat",width=9,height=5,font=("Calibri",20),bg="#DAA520",command=Search)
    b3.pack(side=LEFT,padx=50,pady=5)

    def go_back():
        sub_project1.destroy()
        project.deiconify()
             
    b4 = Button(f5,text = "BACK",relief="flat",width=9,height=5,font=("Calibri",20),bg="#DAA520",command=go_back)
    b4.pack(side=RIGHT,padx=50,pady=5)
    sub_project1.mainloop()
    
b1 = Button(project, text="Camera" ,width=9, height=1,relief="flat", bg="#DAA520",font=("Calibri",20),command=Camera)


b1.pack(in_=top, side=LEFT,pady=5,padx=12)
b2.pack(in_=top, side=LEFT,pady=5,padx=12)






project.mainloop()
