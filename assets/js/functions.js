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
   let iframe = document.querySelector("iframe");
   let msg = JSON.parse(localStorage.msg);
   var icon = document.getElementById('theme-icon');
   if (currentMode == 'dark-poole') {
      document.documentElement.setAttribute('data-theme', 'light-poole');
		window.localStorage.setItem('theme', 'light');
      icon.src = '/assets/images/moon.ico';
		window.localStorage.setItem('data-theme', 'light-poole');
      msg.theme = 'github-light';
   }
   else {
      document.documentElement.setAttribute('data-theme', 'dark-poole');
		window.localStorage.setItem('theme', 'dark');
      icon.src = '/assets/images/sun.ico';
		window.localStorage.setItem('data-theme', 'dark-poole');
      msg.theme = 'photon-dark';
   }
   if (cur_degree == 360) cur_degree = 0;
   else cur_degree += 360;
   icon.style.transform = 'rotate(' + cur_degree + 'deg)';
   icon.style.transition = 'all 1s';
   localStorage.setItem('msg', JSON.stringify(msg));
   iframe.contentWindow.postMessage(msg, "https://utteranc.es");
}

// Hide header on scroll down
var didScroll;
var lastScrollTop = 0;
var delta = 5;
var navbarHeight = document.getElementsByClassName('header')[0].clientHeight;

addEventListener('scroll', (event) => {
  didScroll = true;
});

setInterval(function() {
   if (didScroll) {
      hasScrolled();
      didScroll = false;
   }
}, 250);

function hasScrolled() {
   var st = window.scrollY;
    
   // Make scroll more than delta
   if(Math.abs(lastScrollTop - st) <= delta)
       return;
    
   // If scrolled down and past the navbar, add class .nav-up.
   if (st > lastScrollTop && st > navbarHeight){
      // Scroll Down
      document.getElementsByClassName('header')[0].classList.add('nav-up');
      document.getElementsByClassName('sidebar')[0].style.top = '2.5rem';
   } else {
      // Scroll Up
      if(st + window.innerHeight < document.documentElement.scrollHeight) {
         document.getElementsByClassName('header')[0].classList.remove('nav-up');
         document.getElementsByClassName('sidebar')[0].style.top = '5rem';
      }
   }
   lastScrollTop = st;
}