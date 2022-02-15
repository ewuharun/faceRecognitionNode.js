var express = require('express');
const multer = require('multer');
const script = require('./script')
const fs = require('fs')


const path = require("path");
const canvas = require('canvas');
const faceapi = require('face-api.js');
// const { Canvas, Image, ImageData } = canvas
// faceapi.env.monkeyPatch({ Canvas, Image, ImageData })
const modelPath = path.join(__dirname, './models');


// import faceapi from "face-api.js";

// import * as canvas from 'canvas';
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const faceDetectionNet = faceapi.nets.ssdMobilenetv1;
const minConfidence = 0.5;
const faceDetectionOptions = new faceapi.SsdMobilenetv1Options({ minConfidence });













//start image upload multer setup
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, './uploads/');
      },
    filename: function (req, file, cb) {
        cb(null, file.originalname);
    }
});
const filterImageType = (req,file,cb)=>{
    if(file.mimetype ==='image/jpeg' || file.mimetype==='image/png'){
        cb(null,true);
    }else{
        cb(null,false)
    }
};
const upload = multer({storage: storage,limits:{
    fileSize : 1024*1024*5
},fileFilter: filterImageType});
//end image upload multer setup


var app = express();
app.get('/',function(req,res){
    res.render('index.html')
});

app.post('/upload',upload.single('image'),async (req, res) => {
    if (!req.file) {
        console.log("No file received");
        return res.send({
          success: false
        });
    
      } else {
        console.log('file received');

        await faceDetectionNet.loadFromDisk(modelPath);
        await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
        await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
        
        const referenceImage = await canvas.loadImage('./nid/'+req.body.nid+".jpg")//this image will come from database
        const testImage = await canvas.loadImage('./uploads/'+req.file.originalname)
        const detectionsRef = await faceapi.detectAllFaces(referenceImage, faceDetectionOptions)
        .withFaceLandmarks()
        .withFaceDescriptors()
        const detectionsTest = await faceapi.detectAllFaces(testImage, faceDetectionOptions)
        .withFaceLandmarks()
        .withFaceDescriptors();

    const faceMatcher = new faceapi.FaceMatcher(detectionsRef);

    detectionsTest.forEach(fd => {
        const bestMatch = faceMatcher.findBestMatch(fd.descriptor);

        try {
            fs.unlinkSync('./uploads/'+req.file.originalname)
            //file removed
            res.json(bestMatch)
          } catch(err) {
            console.error(err)
          }

        
    });
      
        // var buffer = fs.readFileSync(req.file.path);
        // script.detect(buffer,req.file.originalname)

        // return res.send({
        //   success: detections
        // })
      }
  })


app.listen(3000, () => {
    console.log("Server running on port 3000");
});