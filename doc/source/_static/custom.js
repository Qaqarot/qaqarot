(function() {
    document.write('<div id="lang-switcher"><a id="lang-en">English</a><a id="lang-ja">Japanese</a></div>');
    var root="https://blueqat.readthedocs.io";
    var langen = document.getElementById("lang-en");
    var langja = document.getElementById("lang-ja");
    var path = location.pathname;
    if(path.indexOf("/en/")>=0){
        langen.className = "round-box box-selected";
        langja.className = "round-box";
        langja.href = root + "/ja/latest/";
    } else {
        langen.className = "round-box";
        langja.className = "round-box box-selected";
        langen.href = root + "/en/latest/";
    }
})();
