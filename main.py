import tempfile
import os
from flask import Flask, request, redirect, send_file
from skimage import io
from skimage.transform import resize
import base64
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.optimizers import SGD

app = Flask(__name__)

main_html = """
<html>
	<head>
		<link rel="stylesheet" type="text/css" href="static/styles.css">
	</head>
	<script>
	  var mousePressed = false;
	  var lastX, lastY;
	  var ctx;

	   function getRndInteger(min, max) {
	    return Math.floor(Math.random() * (max - min) ) + min;
	   }

	  function InitThis() {
	      ctx = document.getElementById('myCanvas').getContext("2d");


	      numero = getRndInteger(0, 10);
	      letra = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"];
	      random = Math.floor(Math.random() * letra.length);
	      aleatorio = letra[random];

	      document.getElementById('mensaje').innerHTML  = 'Escriba en kanji el dia ' + aleatorio;
	      document.getElementById('numero').value = aleatorio;

	      $('#myCanvas').mousedown(function (e) {
		  mousePressed = true;
		  Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
	      });

	      $('#myCanvas').mousemove(function (e) {
		  if (mousePressed) {
		      Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
		  }
	      });

	      $('#myCanvas').mouseup(function (e) {
		  mousePressed = false;
	      });
	  	    $('#myCanvas').mouseleave(function (e) {
		  mousePressed = false;
	      });
	  }

	  function Draw(x, y, isDown) {
	      if (isDown) {
		  ctx.beginPath();
		  ctx.strokeStyle = 'black';
		  ctx.lineWidth = 11;
		  ctx.lineJoin = "round";
		  ctx.moveTo(lastX, lastY);
		  ctx.lineTo(x, y);
		  ctx.closePath();
		  ctx.stroke();
	      }
	      lastX = x; lastY = y;
	  }

	  function clearArea() {
	      ctx.setTransform(1, 0, 0, 1, 0, 0);
	      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
	  }

	  //https://www.askingbox.com/tutorial/send-html5-canvas-as-image-to-server
	  function prepareImg() {
	     var canvas = document.getElementById('myCanvas');
	     document.getElementById('myImage').value = canvas.toDataURL();
	  }



	</script>
	
	<body onload="InitThis();">
	    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>
	    <script type="text/javascript" ></script>
	    <div align="center">
		<h1 id="mensaje">Dibujando...</h1>
		<canvas id="myCanvas" width="200" height="200" style="border:2px solid black"></canvas>
		<br/>
		<br/>
		<div id="botones1">
		      <button onclick="javascript:clearArea();return false;">Borrar</button>
		      <form method="post" action="upload" onsubmit="javascript:prepareImg();"  enctype="multipart/form-data">
			      <input id="numero" name="numero" type="hidden" value="">
			      <input id="myImage" name="myImage" type="hidden" value="">
			      <input id="bt_upload" type="submit" value="Enviar">
		      </form>
	    	</div>
	    	
		<div id="botones2">
			<form method="get" action="prepare" onsubmit="javascript:prepareImg();"  enctype="multipart/form-data">
			      <input id="numero" name="numero" type="hidden" value="">
			      <input id="myImage" name="myImage" type="hidden" value="">
			      <input id="bt_upload" type="submit" value="prepara datos">
		      </form>
		      <form method="get" action="X.npy" onsubmit="javascript:prepareImg();"  enctype="multipart/form-data">
			      <input id="numero" name="numero" type="hidden" value="">
			      <input id="myImage" name="myImage" type="hidden" value="">
			      <input id="bt_upload" type="submit" value="descargar x">
		      </form>
		      <form method="get" action="y.npy" onsubmit="javascript:prepareImg();"  enctype="multipart/form-data">
			      <input id="numero" name="numero" type="hidden" value="">
			      <input id="myImage" name="myImage" type="hidden" value="">
			      <input id="bt_upload" type="submit" value="decargar y">
		      </form>
              <form method="post" action="prediction" onsubmit="javascript:prepareImg();"  enctype="multipart/form-data">
			      <input id="numero" name="numero" type="hidden" value="">
			      <input id="myImage" name="myImage" type="hidden" value="">
			      <input id="bt_upload" type="submit" value="Entrenar y predice">
		      </form>
		</div>
		
		
		
	    </div>
	    
	    <div id="imagen" align="center">
      		<img src="https://marcjapan.files.wordpress.com/2007/02/kanji2x.JPG" width="300"/>
    	    </div>
	</body>
</html>

""" 
@app.route("/prediction", methods=['POST'])
def execute():
    try:
        img_data = request.form.get('myImage').replace("data:image/png;base64,", "")
        aleatorio = request.form.get('numero')
        file_name = aleatorio + '_image.png'
        with tempfile.NamedTemporaryFile(delete=False, mode="w+b", suffix='.png', dir=str(aleatorio)) as fh:
            fh.write(base64.b64decode(img_data))
        print(f"Image uploaded {file_name}")
    except Exception as err:
        print("Error occurred")
        print(err)

    X_raw = np.load('X.npy')
    X_raw = X_raw/255.
    y = np.load('y.npy')
    X = []
    size = (28, 28)
    for x in X_raw:
        X.append(resize(x, size))
    X = np.array(X)
    y_n = np.empty((y.shape[0]))
    day_mapping = {'Lunes': 0, 'Martes': 1, 'Miercoles': 2, 'Jueves': 3, 'Viernes': 4, 'Sabado': 5, 'Domingo': 6}
    for i in range(y.shape[0]):
        if y[i] in day_mapping:
            y_n[i] = day_mapping[y[i]]
    X_train, X_test, y_train, y_test = train_test_split(X, y_n, test_size=0.20, random_state=42, stratify=y)

    if X_train.ndim == 3:
        X_train = X_train[..., None]
        X_test = X_test[..., None]
    bs = 16
    lr = 0.0005
    model = Sequential([Conv2D(32, 3, activation='relu', input_shape=(*size, 1)),
                        MaxPool2D(),
                        Conv2D(64, 3, activation='relu', padding='same'),
                        MaxPool2D(),
                        Conv2D(128, 3, activation='relu', padding='same'),
                        MaxPool2D(),
                        Flatten(),
                        Dense(128, activation='relu'),  # modificar
                        Dense(7, activation='softmax')])  # no modificar

    optimizer1 = SGD(learning_rate=lr)
    model.compile(optimizer=optimizer1, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    log = model.fit(X_train, y_train, batch_size=bs, epochs=400, validation_data=(X_test, y_test))
    
    idx = np.random.choice(X_test.shape[0], 1)[0]
    im = X_test[idx]
    label = y_test[idx]
    
    salida = model.predict(im[None, :, :, :])[0]
    return "se predijo el dato de forma exitosa str(im) -> str(salida.argmax())"

@app.route("/")
def main():
    return(main_html)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        img_data = request.form.get('myImage').replace("data:image/png;base64,", "")
        aleatorio = request.form.get('numero')
        folder_id = '1SAexmaIUn4_SlQUiPLWjDuZcZiSiMuzy'  # Reemplaza con la ID de tu carpeta en Google Drive
        file_name = aleatorio + '_image.png'
        with tempfile.NamedTemporaryFile(delete = False, mode = "w+b", suffix='.png', dir=str(aleatorio)) as fh:
            fh.write(base64.b64decode(img_data))
        print(f"Image uploaded {file_name}")
    except Exception as err:
        print("Error occurred")
        print(err)

    return redirect("/", code=302)
    
@app.route('/prepare', methods=['GET'])
def prepare_dataset():
    images = []
    d = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
    digits = []
    for digit in d:
      filelist = glob.glob('{}/*.png'.format(digit))
      images_read = io.concatenate_images(io.imread_collection(filelist))
      images_read = images_read[:, :, :, 3]
      digits_read = np.array([digit] * images_read.shape[0])
      images.append(images_read)
      digits.append(digits_read)
    images = np.vstack(images)
    digits = np.concatenate(digits)
    np.save('X.npy', images)
    np.save('y.npy', digits)
    return "EL DATA SET HA SIDO PROCESADO :3"

@app.route('/X.npy', methods=['GET'])
def download_X():
    return send_file('./X.npy')
@app.route('/y.npy', methods=['GET'])
def download_y():
    return send_file('./y.npy')
    
if __name__ == "__main__":
    
    digits = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
    for d in digits:
        if not os.path.exists(str(d)):
            os.mkdir(str(d))
    app.run()
