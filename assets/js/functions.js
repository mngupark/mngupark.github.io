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