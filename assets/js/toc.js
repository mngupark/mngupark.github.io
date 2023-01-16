window.addEventListener('DOMContentLoaded', () => {

	const observer = new IntersectionObserver(entries => {
		entries.forEach(entry => {
			const id = encodeURI(entry.target.getAttribute('id'));
            if (entry.intersectionRatio > 0) {
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