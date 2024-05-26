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
        console.log('binded');
        $(document).on('click','.btn-register',App.loginUser);
    },

    loginUser: function(event) {
        event.preventDefault();
        console.log('hi');
        var productInstance;

        var username = document.getElementById('username').value;
        var password = document.getElementById('password').value;
        
        web3.eth.getAccounts(function(error,accounts){

            if(error) {
                console.log(error);
            }

            var account=accounts[0];
            console.log(account);

            App.contracts.product.deployed().then(function(instance){
                productInstance=instance;
                return productInstance.loginUser(web3.fromAscii(username),web3.fromAscii(password),{from:account});
            }).then(function(result){
                console.log(result);
                if(result==true) {
                    window.location.assign('dashboard.html')
                } else{
                    document.getElementById('error').innerHTML='Invalid Credentials';
                }
                // window.location.reload();
                document.getElementById('username').innerHTML='';
                document.getElementById('password').innerHTML='';

            }).catch(function(err){
                console.log(err.message);
            });
        });
    }
}

$(function() {

    $(window).on('load',function() {
        App.init();
    })
})