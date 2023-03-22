class Queue {
	constructor() {
	  	this._arr = [];
	  	this._lastToc = new String();
	}
	enqueue(item) {
	  	this._arr.push(item);
	}
	dequeue() {
	  	return this._arr.shift();
	}
}

const activatedToc = new Queue();
const headings = document.querySelectorAll('h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]');
const sidebarAnchors = document.querySelectorAll('.sidebar li a');
const headingIds = [...headings].map((heading) => heading.id);

const observerCallback = (entries) => {
	entries.forEach(entry => {
		var id = entry.target.id;
        if (entry.isIntersecting == true && entry.intersectionRatio > 0) {
			if (activatedToc._arr.length == 0 && activatedToc._lastToc.length != 0)
				sidebarAnchors[headingIds.indexOf(activatedToc._lastToc)].parentElement.classList.remove('active');
			activatedToc.enqueue(decodeURI(id));
			sidebarAnchors[headingIds.indexOf(id)].parentElement.classList.add('active');
		} else {
			if (activatedToc._arr.find((element) => element == id) != undefined) {
				activatedToc._lastToc = activatedToc.dequeue();
				console.log(activatedToc);
			}
			if (activatedToc._arr.length != 0)
				sidebarAnchors[headingIds.indexOf(id)].parentElement.classList.remove('active');
		}
		// if (headingIds.indexOf(id) == 0 && activatedToc._lastToc.length != 0) {
		// 	activatedToc._lastToc = activatedToc.dequeue();
		// }
	});
};

const observer = new IntersectionObserver(observerCallback);

window.addEventListener('load', () => {
	// Track only headings that have an `id` applied
	headings.forEach((heading) => {
		observer.observe(heading);
	});

});