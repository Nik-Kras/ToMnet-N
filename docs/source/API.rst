API
===

Data Generation
==================

*Map Generation*
-----------------

.. autofunction:: src.data.make_map.generate_wfc_walls
.. autofunction:: src.data.make_map.put_goals_and_player
.. autofunction:: src.data.make_map.generate_map
.. autofunction:: src.data.make_map.create_maps_dataset

*Bot AI*
-----------------

.. autofunction:: src.data.bot_ai.play_map
    
*Game Generation*
-----------------

You can generate games using `src.data.make_dataset.py`
Call it from console and generate N games played by M agents

.. code-block:: console

   (.venv) $ python src.data.make_dataset.py

