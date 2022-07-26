---
layout: default
title: Categories
---

# Categories

Browse all posts by categories.

{% for category in site.categories %}
  <li>
      {% assign category_name = category[0] %}
      <h3>{{ category_name | uppercase }}</h3>
      <ul>
        {% for post in site.categories[category_name] %}
          <li><a href="{{ post.url }}">{{ post.title }}</a></li>
        {% endfor %}
      </ul>
  </li>
{% endfor %}