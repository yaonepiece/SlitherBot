var ws_me = new WebSocket("ws://127.0.0.1:32768/");
var sbtn = document.getElementById(playh).children[0];
var temp,temp2;
ws_me.onmessage = function(e)
{
	window.clearInterval(temp);
	window.clearInterval(temp2);
	temp = setTimeout(()=>
	{
		if(typeof(fpsls)!='undefined' && typeof(snake)!='undefined' && typeof(fmlts)!='undefined' && typeof(snake.sct)!='undefined')
		{
			ws_me.send(Math.floor(15 * (fpsls[snake.sct] + snake.fam / fmlts[snake.sct] - 1) - 5) / 1);
		}else if(e.data == "start"){
			setTimeout(()=>{
				sbtn.onclick();
				setTimeout(()=>{
					ws_me.send(Math.floor(15 * (fpsls[snake.sct] + snake.fam / fmlts[snake.sct] - 1) - 5) / 1);
				},1500);
			},5000);
		}else{
			ws_me.send("stopped");
			setTimeout(()=>{
				sbtn.onclick();
				setTimeout(()=>{
					ws_me.send(Math.floor(15 * (fpsls[snake.sct] + snake.fam / fmlts[snake.sct] - 1) - 5) / 1);
				},1500);
			},10000);
		}
		//console.log(e.data);
	},10);

	temp2 = setTimeout(()=>
	{
		if(typeof(xm)!='undefined' && typeof(ym)!='undefined')
		{
			if(e.data =='0')
			{
				xm = Math.cos(Math.atan2(ym,xm)-0.4)*100;
				ym = Math.sin(Math.atan2(ym,xm)-0.4)*100;
			}else
			{
				xm = Math.cos(Math.atan2(ym,xm)+0.4)*100;
				ym = Math.sin(Math.atan2(ym,xm)+0.4)*100;
			}
			//console.log(Math.atan2(ym,xm) + " " + xm + " "+ ym);
		}
	},10);
	
	
}