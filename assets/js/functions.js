function toggleCategory(id) {
    var e_ul = document.getElementById(id);
    var e_btn = document.getElementById(id + '_btn');
       if(e_ul.style.visibility == 'hidden') {
          e_ul.style.visibility = 'visible';
          e_ul.style.opacity = 1;
          e_ul.style.height = 'auto';
          e_btn.style.backgroundColor = getComputedStyle(document.documentElement).getPropertyValue('--button-hover-bg-color');
       }
       else {
          e_ul.style.visibility = 'hidden';
          e_ul.style.opacity = 0;
          e_ul.style.height = '0';
          e_btn.style.backgroundColor = getComputedStyle(document.documentElement).getPropertyValue('--button-bg-color');
       }
}

let cur_degree = 0;

function toggleTheme() {
   let currentMode = localStorage.getItem('data-theme');
   var icon = document.getElementById('theme-icon');
   if (currentMode == 'dark-poole') {
      document.documentElement.setAttribute('data-theme', 'light-poole');
		window.localStorage.setItem('theme', 'light');
      icon.src = '/assets/images/moon.ico';
		window.localStorage.setItem('data-theme', 'light-poole');
   }
   else {
      document.documentElement.setAttribute('data-theme', 'dark-poole');
		window.localStorage.setItem('theme', 'dark');
      icon.src = '/assets/images/sun.ico';
		window.localStorage.setItem('data-theme', 'dark-poole');
   }
   if (cur_degree == 360) cur_degree = 0;
   else cur_degree += 360;
   icon.style.transform = 'rotate(' + cur_degree + 'deg)';
   icon.style.transition = 'all 1s';
}

window.addEventListener('DOMContentLoaded', () => {

	const observer = new IntersectionObserver(entries => {
		entries.forEach(entry => {
			const id = entry.target.getAttribute('id');
			if (entry.intersectionRatio > 0) {
				document.querySelector(`div li a[href="#${id}"]`).parentElement.classList.add('active');
			} else {
				document.querySelector(`div li a[href="#${id}"]`).parentElement.classList.remove('active');
			}
		});
	});

	// Track all sections that have an `id` applied
	document.querySelectorAll('a[id]').forEach((section) => {
		observer.observe(section);
	});
	
});