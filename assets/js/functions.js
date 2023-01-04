function toggle_category(id) {
    var e_ul = document.getElementById(id);
    var e_btn = document.getElementById(id + '_btn');
       if(e_ul.style.display == 'none') {
          e_ul.style.display = 'block';
          e_btn.style.backgroundColor = getComputedStyle(document.documentElement).getPropertyValue('--button-hover-bg-color');
       }
       else {
          e_ul.style.display = 'none';
          e_btn.style.backgroundColor = getComputedStyle(document.documentElement).getPropertyValue('--button-bg-color');
       }
}

let cur_degree = 0;

function toggle_theme() {
   let currentMode = document.documentElement.getAttribute('data-theme');
   var icon = document.getElementById('theme-icon');
   if (currentMode == 'dark-poole') {
      document.documentElement.setAttribute('data-theme', 'light-poole');
		window.localStorage.setItem('theme', 'light');
      icon.src = '/assets/sun.ico';
		// document.getElementById("theme_btn").innerHTML = "Dark Mode";
   }
   else {
      document.documentElement.setAttribute('data-theme', 'dark-poole');
		window.localStorage.setItem('theme', 'dark');
      icon.src = '/assets/moon.ico';
		// document.getElementById("theme_btn").innerHTML = "Light Mode";
   }
   if (cur_degree == 360) cur_degree = 0;
   else cur_degree += 360;
   icon.style.transform = 'rotate(' + cur_degree + 'deg)';
   icon.style.transition = 'all 1s';
}