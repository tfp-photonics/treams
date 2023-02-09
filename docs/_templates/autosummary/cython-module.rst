{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if members %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
   {% for item in members %}
      {%- if not item.startswith('__') %}
      {{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
