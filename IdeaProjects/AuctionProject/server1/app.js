var express     = require("express"),
     app        = express(),
     bodyParser = require("body-parser"),
     mongoose   = require("mongoose"),
     flash      = require("connect-flash"),
     methodOverride = require("method-override"),
     item = require("./models/item"),
     Comment    = require("./models/comment"),
     User       = require("./models/user"),
     Bid        = require("./models/bid"),
     auctionsetup = require("./models/adminpost"),
     slottable  = require("./models/slotTable")
     seedDB     = require("./seeds"),
     passport   = require("passport"),
     LocalStrategy = require("passport-local"),
     moment       = require("moment");

//requiring Routes
var commentRoutes    = require("./routes/comments"),
    itemRoutes =     require("./routes/items"),
    indexRoutes       = require("./routes/index"),
    bidRoutes         = require("./routes/bids"),
    adminsetupRoutes= require("./routes/adminsetup");
    


mongoose.connect("mongodb+srv://admin:admin@cluster0-oxtvz.mongodb.net/online_auction?retryWrites=true");
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: true}));
app.set("view engine", "ejs");
app.use(express.static(__dirname + "/public"));
app.use(methodOverride("_method"));
app.use(flash());
app.locals.moment = moment;
seedDB(); //seed for the database


//PASSPORT CONFIGURATION
app.use(require("express-session")({
    secret: "i am the dog",
    resave: false,
    saveUninitialized: false
}));

app.use(passport.initialize());
app.use(passport.session());
passport.use(new LocalStrategy(User.authenticate()));
passport.serializeUser(User.serializeUser());
passport.deserializeUser(User.deserializeUser());

app.use(function(req, res, next){
    res.locals.currentUser = req.user;
    res.locals.error = req.flash("error");
    res.locals.success = req.flash("success");
    next();
});

app.use("/",indexRoutes);
app.use("/campgrounds",itemRoutes);
app.use("/auctionsetup", adminsetupRoutes);
app.use("/campgrounds/:id/comments",commentRoutes);
app.use("/campgrounds/:id/bids",bidRoutes);

app.listen("3000", "127.0.0.1", function(){
   console.log("The AuctionService Frontend Server Has Started!");
});