window.addEventListener('DOMContentLoaded', () => {

	const observer = new IntersectionObserver(entries => {
		entries.forEach(entry => {
			const id = entry.target.getAttribute('id');
            if (entry.intersectionRatio > 0) {
                document.querySelector(`.sidebar li a[href="#${id}"]`).parentElement.classList.add('active');
			} else {
				document.querySelector(`.sidebar li a[href="#${id}"]`).parentElement.classList.remove('active');
			}
		});
	});

	// Track only h1 level headers that have an `id` applied
	document.querySelectorAll('h1[id]').forEach((header) => {
		observer.observe(header);
	});
	
});