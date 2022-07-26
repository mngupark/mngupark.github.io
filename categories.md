---
layout: default
title: Categories
---

# Categories

Browse all posts by categories.

{% for categories in site.categories %}
  <h3>{{ category[0] | capitalize }}</h3>
  <ul>
    {% for post in categories[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}