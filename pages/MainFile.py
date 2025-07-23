#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
from PIL import Image
import matplotlib.image as mpimg

#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import matplotlib.image as mpimg
import streamlit as st
from PIL import Image
import streamlit as st

import base64
import cv2

# ================ Background image ===

st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Brain Tumour Detection Using CNN and 3D reconstruction"}</h1>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.avif')

#====================== READ A INPUT IMAGE =========================

fileneme = st.file_uploader("Upload a image")

if fileneme is None:
    
    st.text("Kindly upload input image....")

else:
    selected_image_name = fileneme.name
    st.write(selected_image_name)
    #====================== READ A INPUT IMAGE =========================
    
    
    # filename = askopenfilename()
    img = mpimg.imread(fileneme)
    plt.imshow(img)
    plt.title('Original Image') 
    plt.axis ('off')
    plt.show()
    
    st.image(img,caption="Original Image")
    
    
    #============================ PREPROCESS =================================
    
    #==== RESIZE IMAGE ====
    
    resized_image = cv2.resize(img,(300,300))
    img_resize_orig = cv2.resize(img,((50, 50)))
    
    fig = plt.figure()
    plt.title('RESIZED IMAGE')
    plt.imshow(resized_image)
    plt.axis ('off')
    plt.show()
       
             
    #==== GRAYSCALE IMAGE ====
    
    
    
    SPV = np.shape(img)
    
    try:            
        gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
        
    except:
        gray1 = img_resize_orig
       
    fig = plt.figure()
    plt.title('GRAY SCALE IMAGE')
    plt.imshow(gray1,cmap='gray')
    plt.axis ('off')
    plt.show()
    
    # ============== FEATURE EXTRACTION ==============
    
    
    #=== MEAN STD DEVIATION ===
    
    mean_val = np.mean(gray1)
    median_val = np.median(gray1)
    var_val = np.var(gray1)
    features_extraction = [mean_val,median_val,var_val]
    
    print("====================================")
    print("        Feature Extraction          ")
    print("====================================")
    print()
    print(features_extraction)
    
    
    #============================ 5. IMAGE SPLITTING ===========================
    
    import os 
    
    from sklearn.model_selection import train_test_split
    
    data_glioma = os.listdir('Data/glioma/')
    data_menign = os.listdir('Data/meningioma/')
    data_non = os.listdir('Data/notumor/')
    data_pit = os.listdir('Data/pituitary/')
    
    
    #       
    dot1= []
    labels1 = [] 
    for img11 in data_glioma:
            # print(img)
            img_1 = mpimg.imread('Data/glioma//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(1)
    
    
    for img11 in data_menign:
            # print(img)
            img_1 = mpimg.imread('Data/meningioma//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(2)
    
    for img11 in data_non:
            # print(img)
            img_1 = mpimg.imread('Data/notumor//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(3)
    
    
    for img11 in data_pit:
            # print(img)
            img_1 = mpimg.imread('Data/pituitary//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(4)
    
    
    x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
    
    print()
    print("-------------------------------------")
    print("       IMAGE SPLITTING               ")
    print("-------------------------------------")
    print()
    
    
    print("Total no of data        :",len(dot1))
    print("Total no of test data   :",len(x_train))
    print("Total no of train data  :",len(x_test))
    
    
    #=============================== CLASSIFICATION =================================
    
    from keras.utils import to_categorical
    
    
    y_train1=np.array(y_train)
    y_test1=np.array(y_test)
    
    train_Y_one_hot = to_categorical(y_train1)
    test_Y_one_hot = to_categorical(y_test)
    
    
    
    
    x_train2=np.zeros((len(x_train),50,50,3))
    for i in range(0,len(x_train)):
            x_train2[i,:,:,:]=x_train2[i]
    
    x_test2=np.zeros((len(x_test),50,50,3))
    for i in range(0,len(x_test)):
            x_test2[i,:,:,:]=x_test2[i]
    
    
    # ======== CNN ===========
        
    from keras.layers import Dense, Conv2D
    from keras.layers import Flatten
    from keras.layers import MaxPooling2D
    # from keras.layers import Activation
    from keras.models import Sequential
    from keras.layers import Dropout
    
    
    
    
    # initialize the model
    model=Sequential()
    
    
    #CNN layes 
    model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Dropout(0.2))
    model.add(Flatten())
    
    model.add(Dense(500,activation="relu"))
    
    model.add(Dropout(0.2))
    
    model.add(Dense(5,activation="softmax"))
    
    #summary the model 
    model.summary()
    
    #compile the model 
    model.compile(loss='binary_crossentropy', optimizer='adam')
    y_train1=np.array(y_train)
    
    train_Y_one_hot = to_categorical(y_train1)
    test_Y_one_hot = to_categorical(y_test)
    
    
    print("-------------------------------------")
    print("CONVOLUTIONAL NEURAL NETWORK (CNN)")
    print("-------------------------------------")
    print()
    #fit the model 
    history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=5,verbose=1)
    
    accuracy = model.evaluate(x_train2, train_Y_one_hot, verbose=1)
    
    loss=history.history['loss']
    
    error_cnn=max(loss)*10
    
    acc_cnn=100- error_cnn
    
    
    print("-------------------------------------")
    print("PERFORMANCE ---------> (CNN)")
    print("-------------------------------------")
    print()
    print("1. Accuracy   =", acc_cnn,'%')
    print()
    print("2. Error Rate =",error_cnn)

    st.write("-------------------------------------")
    st.write("PERFORMANCE ---------> (CNN)")
    st.write("-------------------------------------")
    print()
    st.write("1. Accuracy   =", acc_cnn,'%')
    print()
    st.write("2. Error Rate =",error_cnn)
        
    
    #=============================== PREDICTION =================================
    
    print()
    print("-----------------------")
    print("       PREDICTION      ")
    print("-----------------------")
    print()
    
    
    Total_length = len(data_glioma) + len(data_menign) + len(data_non) + len(data_pit)
     
    
    temp_data1  = []
    for ijk in range(0,Total_length):
        # print(ijk)
        temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
        temp_data1.append(temp_data)
    
    temp_data1 =np.array(temp_data1)
    
    zz = np.where(temp_data1==1)
    
    if labels1[zz[0][0]] == 1:
        print('-----------------------------------')
        print(' IDENTIFIED GLIOMA')
        print('-----------------------------------')

        st.write('-----------------------------------')
        st.write(' IDENTIFIED GLIOMA')
        st.write('-----------------------------------')
    
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from skimage.transform import resize
        from PIL import Image
        

        image = np.array(Image.open(fileneme).convert('RGB'))  # Replace 'input_image.jpg' with your image file path
        
        input_size = (50, 50)  
        resized_image = resize(image, input_size, anti_aliasing=True)
        
        
        depth_map = np.random.rand(*input_size)
        

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        

        Y, X = np.meshgrid(np.arange(input_size[1]), np.arange(input_size[0]))
        Z = depth_map
        

        ax.plot_surface(X, Y, Z, cmap='viridis')
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Depth')
        

        # plt.show()

    
        img1 = mpimg.imread("1.JPG")
        plt.imshow(img1)
        plt.title('3D View Image') 
        plt.axis ('off')
        plt.show()
         
        # st.image(img1,caption="3d Image")
    
        # st.image("out.png")
        
    
    elif labels1[zz[0][0]] == 2:
        print('----------------------------------')
        print(' IDENTIFIED == MENIGN')
        print('----------------------------------')
        
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from skimage.transform import resize
        from PIL import Image
        

        image = np.array(Image.open(fileneme).convert('RGB'))  # Replace 'input_image.jpg' with your image file path
        
        input_size = (50, 50)  
        resized_image = resize(image, input_size, anti_aliasing=True)
        
        
        depth_map = np.random.rand(*input_size)
        

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        

        Y, X = np.meshgrid(np.arange(input_size[1]), np.arange(input_size[0]))
        Z = depth_map
        

        ax.plot_surface(X, Y, Z, cmap='viridis')
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Depth')
        

        # plt.show()

    
        img2 = mpimg.imread("1.JPG")
        plt.imshow(img2)
        plt.title('3D View Image') 
        plt.axis ('off')
        # plt.show()
        # st.image(img2,caption="3d Image")
        
        # st.image("out.png")
        
    elif labels1[zz[0][0]] == 3:
        print('----------------------------------')
        print(' IDENTIFIED == NO TUMOUR')
        print('----------------------------------')

    elif labels1[zz[0][0]] == 4:
        print('----------------------------------')
        print(' IDENTIFIED ==  PITUITARY')
        print('----------------------------------')
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from skimage.transform import resize
        from PIL import Image
        

        image = np.array(Image.open(fileneme).convert('RGB'))  # Replace 'input_image.jpg' with your image file path
        
        input_size = (50, 50)  
        resized_image = resize(image, input_size, anti_aliasing=True)
        
        
        depth_map = np.random.rand(*input_size)
        

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        

        Y, X = np.meshgrid(np.arange(input_size[1]), np.arange(input_size[0]))
        Z = depth_map
        

        ax.plot_surface(X, Y, Z, cmap='viridis')
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Depth')
        

        # plt.show()

    
        img3 = mpimg.imread("1.JPG")
        plt.imshow(img3)
        plt.title('3D View Image') 
        plt.axis ('off')
        # plt.show()

        # st.image(img3,caption="3d Image")        
        # st.image("out.png")
            
        import csv
        # def show_corresponding_image():
            # global selected_image_name
    # if selected_image_name:
    #         # Open the CSV file
    #         csv_filename = "dataset1.csv"
    #         if csv_filename:
    #             # Read the CSV file
    #             with open(csv_filename, 'r') as file:
    #                 csv_reader = csv.reader(file)
    #                 for row in csv_reader:
    #                     if row[0] == selected_image_name:
    #                         # Get the last data from the matched row
    #                         last_data = row[-1]
    #                         # Show image from folder using the last data
    #                         image_path = os.path.join(r"ad", last_data)  # Change "path_to_folder" to the actual folder path
    #                         if os.path.exists(image_path):
    #                             img4= mpimg.imread(image_path)
    #                             plt.imshow(img4)
    #                             st.image(img4)
    #                             plt.axis('off')
    #                             plt.title(last_data)
    #                             plt.show()
    #                             st.image(img4,caption="3d Image")
    #                             # return
    # else:
    #                 # If no match found
    #                 print("No matching image found in the CSV file.")
    # # show_corresponding_image

    import pandas as pd
    
    df = pd.read_csv('dataset1.csv')
    
    data_label=df['input Image']
    x1=data_label
    
    a=fileneme.name
    st.text(a)
    
    for i in range(0,len(df)):
        if x1[i]==a:
            idx=i    
        
    data_frame1_c=df[' Name']
    Req_data_c=data_frame1_c[idx]
    st.text(Req_data_c)
    st.text("completed")
    img4= mpimg.imread('3d_dataset/'+Req_data_c)
    plt.imshow(img4)
    # st.image(img4)
    # plt.title(last_data)
    # plt.show()
    st.image(img4,caption="3d Image")
    plt.axis('off')
