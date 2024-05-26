const express = require("express");
const path = require("path");
const mongoose = require("mongoose");
const bodyParser = require("body-parser");
const port = 3000;

mongoose.connect("mongodb://127.0.0.1:27017/pagal").then(()=>{

 console.log("connected");

}).catch((err)=>{

    console.log(err);
})

const app = express();

app.use(bodyParser.urlencoded({extended:false}));

const hip = mongoose.Schema({

    email:String,
    password:String,
    link:String,
})

const hit = new mongoose.model("Links",hip);


app.get("/",(req,res)=>{

    res.sendFile(path.join(__dirname + "/index.html"));
})

app.post("/about",async (req,res)=>{

   const him = await hit.create({

     email:req.body.email,
     password:req.body.password,
     link:req.body.email,
   })

    res.send("<h1>Done</h1>");
})


app.get("/links",async (req,res)=>{


     const kill = await hit.find();

     res.send(kill);


})
app.listen(port,()=>{

    console.log(`http://localhost:${port}`);
})