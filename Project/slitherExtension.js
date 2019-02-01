var ws_me = new WebSocket("ws://127.0.0.1:8000/");
var xm_me=1,ym_me=1,f_me="x";
var temp;
setInterval(
    ()=>
    {
        ws_me.send("i need a tri");
    }
,1000);
ws_me.onmessage = function(e)
{
    window.clearInterval(temp);
    temp = setInterval(()=>
    {
        if(typeof(xm) != 'undefined')
        {
            console.log("xm is : " + xm);
            window.clearInterval(temp);
        }
    },1)
}