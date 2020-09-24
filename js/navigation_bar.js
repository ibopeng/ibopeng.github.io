function navDropdrown() {
	var x = document.getElementById("myTopnav");
	if (x.className === "topnav") {
		x.className += " responsive";
		} else {
		x.className = "topnav";
	}

	var y = document.getElementById("menu_icon");
	if (x.className === "topnav") {
		y.innerHTML = "&#xe5d2;"
	} else {
		y.innerHTML = "&#xe5cd;"
	}
}