import * as THREE from "three";
import { Tween, Easing } from "@tweenjs/tween.js";
import { createUnitMesh } from "./create";
import { getProvincePosition } from "../map/utils";
import { gameState } from "../gameState";
import type { UnitOrder } from "../types/unitOrders";
import { logger } from "../logger";
import { config } from "../config"; // Assuming config is defined in a separate file
import { PowerENUM, ProvinceENUM } from "../types/map";
import { UnitTypeENUM } from "../types/units";
import { sineWave, getTimeInSeconds } from "../utils/timing";

function getUnit(unitOrder: UnitOrder, power: string) {
  if (power === undefined) throw new Error("Must pass the power argument, cannot be undefined")
  let posUnits = gameState.unitMeshes.filter((unit) => {
    return (
      unit.userData.province === unitOrder.unit.origin &&
      unit.userData.type === unitOrder.unit.type &&
      unit.userData.power === power &&
      (unit.userData.isAnimating === false || unit.userData.isAnimating === undefined)
    );
  });

  if (posUnits.length === 0) {
    return -1;
  }

  // Return the first matching unit
  return gameState.unitMeshes.indexOf(posUnits[0]);
}

/* Return a tween animation for the spawning of a unit.
 *  Intended to be invoked before the unit is added to the scene
*/
function createSpawnAnimation(newUnitMesh: THREE.Group): Tween {
  // Start the unit really high, and lower it to the board.
  newUnitMesh.position.setY(1000)
  return new Tween({ y: 1000 })
    .to({ y: 10 }, config.effectiveAnimationDuration || 1000)
    .easing(Easing.Quadratic.Out)
    .onUpdate((object) => {
      newUnitMesh.position.setY(object.y)
    }).start()
}

function createMoveAnimation(unitMesh: THREE.Group, orderDestination: ProvinceENUM): Tween {
  let destinationVector = getProvincePosition(orderDestination);
  if (!destinationVector) {
    throw new Error("Unable to find the vector for province with name " + orderDestination)
  }
  unitMesh.userData.province = orderDestination;
  unitMesh.userData.isAnimating = true
  
  // Store animation start time for consistent wave motion
  const animStartTime = getTimeInSeconds();
  
  let anim = new Tween(unitMesh.position)
    .to({
      x: destinationVector.x,
      y: 10,
      z: destinationVector.z
    }, config.effectiveAnimationDuration)
    .easing(Easing.Quadratic.InOut)
    .onUpdate(() => {
      // Use elapsed time from animation start for consistent wave motion
      const elapsedTime = getTimeInSeconds() - animStartTime;
      unitMesh.position.y = 10 + sineWave(config.animation.unitBobFrequency, elapsedTime, 2); // 2 units amplitude
      if (unitMesh.userData.type === 'F') {
        unitMesh.rotation.z = sineWave(config.animation.fleetRollFrequency, elapsedTime, 0.1);
        unitMesh.rotation.x = sineWave(config.animation.fleetPitchFrequency, elapsedTime, 0.1);
      }
    })
    .onComplete(() => {
      unitMesh.position.y = 10;
      if (unitMesh.userData.type === 'F') {
        unitMesh.rotation.z = 0;
        unitMesh.rotation.x = 0;
      }
      unitMesh.userData.isAnimating = false
    })
    .start();
  gameState.unitAnimations.push(anim);
  return anim
}

/**
 * Creates animations for unit movements based on orders from the previous phase
 *
**/
export function createAnimationsForNextPhase() {
  let previousPhase = gameState.gameData?.phases[gameState.phaseIndex == 0 ? 0 : gameState.phaseIndex - 1]

  // Safety check - if no previous phase or no orders, return
  if (!previousPhase) {
    logger.log("No previous phase to animate");
    return;
  }
  for (const [power, orders] of Object.entries(previousPhase.orders)) {
    if (orders === null) {
      continue
    }
    for (const order of orders) {
      // Check if unit bounced
      // With new format: {A: {"BUD": [results]}, F: {"BUD": [results]}}
      const unitType = order.unit.type;
      const unitOrigin = order.unit.origin;

      let result = undefined;
      if (previousPhase.results && previousPhase.results[unitType] && previousPhase.results[unitType][unitOrigin]) {
        const resultArray = previousPhase.results[unitType][unitOrigin];
        result = resultArray.length > 0 ? resultArray[0] : null;

      }

      if (result === undefined) {
        throw new Error(`No result present in current phase for previous phase order: ${unitType} ${unitOrigin}. Cannot continue`);
      }

      if (result === "bounce") {
        order.type = "bounce"
      }
      // If the result is void, that means the move was not valid?
      if (result === "void") continue;
      let unitIndex = -1

      unitIndex = getUnit(order, power);
      switch (order.type) {
        case "move":
          if (!order.destination) throw new Error("Move order with no destination, cannot complete move.")
          if (unitIndex < 0) throw new Error("Unable to find unit for order " + order.raw)
          // Create a tween for smooth movement
          createMoveAnimation(gameState.unitMeshes[unitIndex], order.destination as keyof typeof ProvinceENUM)
          break;

        case "disband":
          // TODO: Death animation
          if (unitIndex < 0) throw new Error("Unable to find unit for order " + order.raw)
          gameState.scene.remove(gameState.unitMeshes[unitIndex]);
          gameState.unitMeshes.splice(unitIndex, 1);
          break;

        case "build":
          // TODO: Spawn animation?
          let newUnit = createUnitMesh({
            power: PowerENUM[power as keyof typeof PowerENUM],
            type: UnitTypeENUM[order.unit.type as keyof typeof UnitTypeENUM],
            province: order.unit.origin
          })
          gameState.unitAnimations.push(createSpawnAnimation(newUnit))
          gameState.scene.add(newUnit)
          gameState.unitMeshes.push(newUnit)
          break;

        case "bounce":
          // TODO: implement bounce animation
          break;
        case "hold":
          //TODO: Hold animation, maybe a sheild or something?
          break;

        case "convoy":
          // The unit doesn't move, so no animation for now
          break;

        case "retreat":
          if (unitIndex < 0) throw new Error("Unable to find unit for order " + order.raw)
          createMoveAnimation(gameState.unitMeshes[unitIndex], order.destination as keyof typeof ProvinceENUM)
          break;

        case "support":
          break

        default:
          // FIXME: There is an issue where some F are not getting disbanded when I believe they should
          //    check ROM in game 0, turn 2-5.  
          throw new Error(`Unhandled order.type ${order.type}.`)
      }
    }
  }
}
