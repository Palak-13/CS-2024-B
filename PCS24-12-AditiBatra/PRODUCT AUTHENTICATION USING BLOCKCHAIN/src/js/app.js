
App = {
    web3Provider: null,
    contracts: {},

    init: async function() {
        return await App.initWeb3();
    },

    initWeb3: function() {
        if(window.web3) {
            App.web3Provider=window.web3.currentProvider;
        } else {
            App.web3Provider=new Web3.proviers.HttpProvider('http://localhost:8545');
        }

        web3 = new Web3(App.web3Provider);
        return App.initContract();
    },

    initContract: function() {

        $.getJSON('product.json',function(data){

            var productArtifact=data;
            App.contracts.product=TruffleContract(productArtifact);
            App.contracts.product.setProvider(App.web3Provider);
        });

        return App.bindEvents();
    },

    bindEvents: function() {

        $(document).on('click','.btn-register',App.registerProduct);
    },

    registerProduct: function(event) {
        event.preventDefault();

        var productInstance;

        var productId = document.getElementById('productId').value;
        var pOwner = document.getElementById('pOwner').value;  
        var options = {
            text: productId
        };
        
        // Create QRCode Object
        
        
        web3.eth.getAccounts(function(error,accounts){

            if(error) {
                console.log(error);
            }

            var account=accounts[0];
            console.log(account);

            App.contracts.product.deployed().then(function(instance){
                productInstance=instance;
                return productInstance.setProduct(web3.fromAscii(productId),web3.fromAscii(pOwner),{from:account});
            }).then(function(result){
                console.log(result);

                var typeNumber = 4;  
                var errorCorrectionLevel = 'L';  
                var qr = qrcode(typeNumber, errorCorrectionLevel);  
                var inputText = productId;  
                qr.addData(inputText);  
                qr.make();  
                document.getElementById('qrcode').innerHTML = qr.createImgTag();  
                document.getElementById('productId').innerHTML='';
                document.getElementById('pOwner').innerHTML='';

            }).catch(function(err){
                console.log(err.message);
            });
        });
    }
};

$(function() {

    $(window).load(function() {
        App.init();
    })
})