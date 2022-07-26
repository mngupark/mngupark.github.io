---
layout: default
title: Categories
---

# Categories

Browse all posts by categories.

{% for category in site.categories %}
  <h2>{{ category | first }}</h2>
  <h3>{{ category[0] | capfirst }}</h3>
  <ul>
    {% for post in category[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}