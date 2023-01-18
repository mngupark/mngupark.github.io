window.addEventListener('DOMContentLoaded', () => {

	const observer = new IntersectionObserver(entries => {
		entries.forEach(entry => {
			var id = encodeURI(entry.target.getAttribute('id'));
			if (document.querySelector(`.sidebar li a[href="#${id}"]`) == null)
				id = decodeURI(id);
            if (entry.intersectionRatio > 0) {
				console.log(document.querySelector(`.sidebar li a[href="#${id}"]`));
                document.querySelector(`.sidebar li a[href="#${id}"]`).parentElement.classList.add('active');
			} else {
				document.querySelector(`.sidebar li a[href="#${id}"]`).parentElement.classList.remove('active');
			}
		});
	});

	// Track only headings that have an `id` applied
	document.querySelectorAll('h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]').forEach((heading) => {
		observer.observe(heading);
	});
	
});