import 'package:flutter/material.dart';

class ParkingSpaceWidget extends StatelessWidget {
  final String id;
  final bool occupied;
  final bool selected;
  final VoidCallback? onTap;

  const ParkingSpaceWidget({
    super.key,
    required this.id,
    required this.occupied,
    this.selected = false,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: occupied ? Colors.red.shade400 : Colors.green.shade400,
      borderRadius: BorderRadius.circular(8),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(8),
        child: Container(
          margin: const EdgeInsets.all(2),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
              color: selected ? Colors.white : Colors.black26,
              width: selected ? 3 : 1,
            ),
          ),
          child: Center(
            child: Text(
              id,
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 18,
              ),
            ),
          ),
        ),
      ),
    );
  }
}
